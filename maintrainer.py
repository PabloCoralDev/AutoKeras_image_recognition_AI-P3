#!/usr/bin/env python3
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------- basic setup --------
TRAIN_IMAGES_DIR = Path("images")
LABELS_CSV = Path("labels.csv")
PREDICT_IMAGE = Path("predict_image/current.jpg")  # optional; change to your file or set to None
OUTPUT_DIR = Path("results")

IMG_SIZE = 224
BATCH = 32
VAL_SPLIT = 0.2
EPOCHS = 10
TRIALS = 3
SEED = 42

# ensure AutoKeras preprocessors can be reloaded (handles Keras 3)
try:
    from autokeras.preprocessors import common as _ak_common  # type: ignore
    from keras.saving import register_keras_serializable as _register

    for _name in ("AddOneDimension", "CastToString", "ExpandLastDim"):
        _proc = getattr(_ak_common, _name, None)
        if _proc is not None:
            try:
                _register(package="autokeras")(_proc)
            except Exception:
                pass
except Exception:
    pass


def read_labels():
    if not TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Training images folder not found: {TRAIN_IMAGES_DIR}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

    items = []
    with open(LABELS_CSV, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            filename, label = row[0].strip(), row[1].strip().strip("[] ")
            path = (TRAIN_IMAGES_DIR / filename).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Image listed in CSV not found: {filename}")
            items.append({"filepath": str(path), "label": label})
    if not items:
        raise RuntimeError("No training samples found.")
    return items


def stratified_split(items):
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for item in items:
        by_label[item["label"]].append(item)

    train, val = [], []
    for label, group in by_label.items():
        if len(group) == 1:
            train.extend(group)
            continue
        k = max(1, int(round(VAL_SPLIT * len(group))))
        rng.shuffle(group)
        val.extend(group[:k])
        train.extend(group[k:])
    return train, val


def load_predict_image():
    if PREDICT_IMAGE is None:
        return None
    image_path = Path(PREDICT_IMAGE)
    if not image_path.exists():
        print(f"[WARN] Prediction image not found: {image_path}. Skipping prediction.")
        return None
    img = tf.io.read_file(str(image_path))
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0), image_path.name


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def train_and_predict():
    items = read_labels()
    classes = sorted({item["label"] for item in items}, key=lambda x: x.lower())
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    for item in items:
        item["y"] = class_to_id[item["label"]]

    train_items, val_items = stratified_split(items)

    def dataset(records, shuffle):
        paths = [r["filepath"] for r in records]
        labels = [r["y"] for r in records]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def load(path, label):
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=min(1000, len(paths)), seed=SEED, reshuffle_each_iteration=True)
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    train_ds = dataset(train_items, shuffle=True)
    val_ds = dataset(val_items, shuffle=False)

    ensure_dir(OUTPUT_DIR)

    try:
        import autokeras as ak
    except ImportError:
        raise SystemExit("Install AutoKeras: pip install autokeras tensorflow numpy")

    input_node = ak.ImageInput()
    output_node = ak.ImageBlock(
        block_type="vanilla",
        normalize=True,
        augment=False,
    )(input_node)
    output_node = ak.ClassificationHead()(output_node)

    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        max_trials=TRIALS,
        seed=SEED,
        directory=str(OUTPUT_DIR),
        project_name="ak_project",
    )

    clf.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

    model = clf.export_model()
    model_path = OUTPUT_DIR / "autokeras_image_classifier.keras"
    model.save(model_path)
    print(f"[INFO] Saved model: {model_path}")

    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    prediction_input = load_predict_image()
    if prediction_input is None:
        return

    tensor, name = prediction_input
    probs = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    print(f"[PREDICT] {name}: {classes[idx]} ({probs[idx]:.2%})")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    train_and_predict()


if __name__ == "__main__":
    main()

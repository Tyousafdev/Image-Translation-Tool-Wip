import os, cv2, argparse, xml.etree.ElementTree as ET, random, shutil
from pathlib import Path

def crop_and_save(img_path, bbox, out_dir, fname):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Could not read {img_path}")
        return None
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / fname
    cv2.imwrite(str(out_path), crop)
    return out_path

def build_from_xml(xml_root, img_root, out_dir, label, node_type="text", limit=None):
    xml_root = Path(xml_root)
    tmp_dir = Path(out_dir) / f"_tmp_{label}"
    os.makedirs(tmp_dir, exist_ok=True)
    count = 0

    for xml_file in xml_root.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # strip namespace if present
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        book_title = root.attrib.get("title")
        img_dir = Path(img_root) / book_title
        if not img_dir.exists():
            continue

        for page in root.findall(".//page"):
            page_index = int(page.attrib["index"])
            img_path = img_dir / f"{page_index:03d}.jpg"
            if not img_path.exists():
                continue

            for node in page.findall(node_type):
                if node_type == "onomatopoeia":
                    xs = [int(node.attrib[k]) for k in node.attrib if k.startswith("x")]
                    ys = [int(node.attrib[k]) for k in node.attrib if k.startswith("y")]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                else:
                    xmin = int(node.attrib["xmin"])
                    ymin = int(node.attrib["ymin"])
                    xmax = int(node.attrib["xmax"])
                    ymax = int(node.attrib["ymax"])

                out_file = f"{book_title}_{page_index:03d}_{node.attrib.get('id','none')}.png"
                crop_and_save(img_path, (xmin, ymin, xmax, ymax), tmp_dir, out_file)
                count += 1
                if limit and count >= limit:
                    print(f"[{label}] stopped at limit {limit}")
                    return tmp_dir
    print(f"[{label}] saved {count} crops.")
    return tmp_dir

def split_train_val(tmp_dir, out_dir, label, split_ratio=0.8):
    tmp_dir = Path(tmp_dir)
    files = list(tmp_dir.glob("*.png"))
    random.shuffle(files)
    split_idx = int(len(files) * split_ratio)

    train_dir = Path(out_dir) / "train" / label
    val_dir = Path(out_dir) / "val" / label
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for f in files[:split_idx]:
        shutil.move(str(f), train_dir / f.name)
    for f in files[split_idx:]:
        shutil.move(str(f), val_dir / f.name)

    print(f"[{label}] train={split_idx}, val={len(files)-split_idx}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Path to Manga109/images")
    ap.add_argument("--dialogs", required=True, help="Path to Manga109/annotations")
    ap.add_argument("--sfx", required=True, help="Path to Manga109/annotations_COO")
    ap.add_argument("--out", default="dataset", help="Output dataset root")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit per class")
    args = ap.parse_args()

    # build crops
    tmp_bubble = build_from_xml(args.dialogs, args.images, args.out, "bubble", node_type="text", limit=args.limit)
    tmp_sfx = build_from_xml(args.sfx, args.images, args.out, "sfx", node_type="onomatopoeia", limit=args.limit)

    # split into train/val
    if tmp_bubble: split_train_val(tmp_bubble, args.out, "bubble")
    if tmp_sfx: split_train_val(tmp_sfx, args.out, "sfx")

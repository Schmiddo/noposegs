#!/usr/bin/env python

import os
from pathlib import Path

import tyro

@tyro.cli
def rename_llff(dir: Path):
    # creates hardlinks to the downsized images that have the right name.
    src_image_filenames = sorted([f.name for f in dir.glob("images/*") if f.name[-3:].lower() == "jpg"])
    for res in (2, 4, 8):
        # yes, we rename PNGs to JPGs. So far nobody died.
        tgt_image_files = sorted(dir.glob(f"images_{res}/*.png"))
        for n, f in zip(src_image_filenames, tgt_image_files):
            new_path = (dir / f"images_{res}" / n)
            new_path.hardlink_to(f)
            #print(new_path, f)
            #print(f.absolute(), f"{f.parent.name}/{n}")
            #os.link(f"{f.parent.name}/{n}", f.absolute())


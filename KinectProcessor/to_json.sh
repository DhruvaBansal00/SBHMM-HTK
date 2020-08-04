#/bin/sh


# SEARCH_FOLDER="*"

for f in $(find /home/dhruva/Desktop/CopyCat/Media/ -name '*.mkv'); do ./offline_processor $f "${f%.*}.json"; done
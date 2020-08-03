#/bin/sh


# SEARCH_FOLDER="*"

for f in $(find real5-mkv/ -name '*.mkv'); do ./offline_processor $f "${f%.*}.json"; done
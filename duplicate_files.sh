n=10000
workdir=$(dirname $0)
filedir=${workdir}/data/256/images/25/256_1/train/images
all_files=$(ls ${filedir}/*)

for i in {1..19..1}; do
    for f in ${all_files};  do
        filename=$(basename "$f")
        filename="${filename%.*}"
        filename=$(echo $filename | sed 's/[0-9]*$//' | sed 's/_$//')
        ext="${f##*.}"

        while [[ "${filename}${n}.${ext}" == "$(basename "$f")" ]]; do
            n=$((n+1))
        done
        
        cp -p "$f" ${filedir}/"$filename"${n}.${ext}
        n=$((n+1))
    done
done

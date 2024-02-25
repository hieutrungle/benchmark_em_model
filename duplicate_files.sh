n=10000
workdir=$(dirname $0)
filedir=${workdir}/tmp

for f in ${filedir}/*;  do
    filename=$(basename "$f")
    filename="${filename%.*}"
    ext="${f##*.}"
    cp -p "$f" ${filedir}/"$filename"_${n}.${ext}
    n=$((n+1))
done
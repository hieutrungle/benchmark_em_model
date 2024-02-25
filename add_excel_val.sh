FILE=$1

NAMES=`cut -d',' -f 1 $FILE | sort -u`

for NAME in $NAMES; do
    echo -n "$NAME" 
    # VALUES=`grep "$NAME" $FILE | cut -d',' -f2`
    # for VAL in $VALUES; do
    #     echo -n ",$VAL"
    # done
    # echo ""
done
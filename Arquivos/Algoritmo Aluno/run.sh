for i in {2..5}
do
    for FILE in csv/*.csv; 
    do 
        echo $FILE not_save random $i;
        python3 method.py $FILE not_save random $i
    done
done

for i in {1..5}
do
    for FILE in csv/*.csv; 
    do 
        echo $FILE save_concur sorted $i;
        python3 method.py $FILE save_concur sorted $i
    done
done

for i in {1..5}
do
    for FILE in csv/*.csv; 
    do 
        echo $FILE $i;
        python3 random_K2.py $FILE $i
    done
done
for d in $1/*; do
  if [ -d "$d" ]; then
    base=`basename $d`
    python compare_result_files.py --input-directory $d --qualifier $base    
  fi
done


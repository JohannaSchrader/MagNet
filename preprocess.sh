#!/usr/bin/env bash

in_file=$1  # path to GSE211692_RAW.tar
working_dir=$2
mapping_file=$3  # path to mapping_file.txt

if [ ! -f "${in_file}" ]; then
    echo "${in_file} not found."
    exit 1
fi

if [ ! -f "${mapping_file}" ]; then
    echo "${mapping_file} not found."
    exit 1
fi

if [ -d "${working_dir}" ]; then
    rm -rf "${working_dir}"
fi
mkdir -p "${working_dir}"

echo "Extracting files"
tar xvf "${in_file}" -C "${working_dir}"
find "${working_dir}" -type f -name "*.txt.gz" -exec gunzip {} \;

dict_file="rename_mapping.txt"
> "${dict_file}"

echo "Renaming and building dictionary"
for file in "${working_dir}"/*.txt; do
    [ -e "$file" ] || continue
    basename=$(basename "${file}")
    id_part=$(echo "${basename}" | cut -d '_' -f 1)
    renamed_basename=$(echo "${basename}" | cut -d '_' -f 2-)
    dict_key="${renamed_basename%.txt}"
    echo "${dict_key}:${id_part}" >> "${dict_file}"
    mv "${file}" "${working_dir}/${renamed_basename}"
done

echo "Sorting files into subfolders"
while IFS= read -r dest_path; do
    [ -z "$dest_path" ] && continue
    
    current_name=$(basename "$dest_path")
    src="${working_dir}/${current_name}"
    dst="${working_dir}/${dest_path}"
    
    mkdir -p "$(dirname "$dst")"
    
    if [ -f "$src" ]; then
        mv "$src" "$dst"
    else
        echo "Warning: Expected file $src not found. Skipping."
    fi
done < "${mapping_file}"

# Run Python Scripts
root_dir=$(cd "$(dirname "$0")"; pwd)
PYTHONPATH=$PYTHONPATH:. python backgroundcorrect.py -i "${working_dir}" 
PYTHONPATH=$PYTHONPATH:. python sort_data.py 

# Clean up
rm -rf "${working_dir}"
echo 'done'
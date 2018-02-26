nvccPresence=$(which nvcc)

if [[ $nvccPresence = *nvcc* ]]; then
  echo "System has nvcc installed, compiling GPU version"
  make gpubuild
else
  echo "Compiling standard CPU version."
  make
fi

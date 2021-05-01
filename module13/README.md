# Image Filtering

## Execute
Use the included [run.sh](run.sh) file to compile and execute the code

The script can accept arguments:

- `$1` is the number of sequences to run in the test harness
  - ex `./run.sh 10` will execute 10 runs after compiling the code
- `$2` is a flag
  - ex `./run.sh 10 1` will execute 10 runs and delete the intermediate filtered files
  - This is useful for large runs
- Defaults:
  - `$1` defaults to 5
  - `$2` the delete flag defaults to 0, which won't delete any bitmaps from intermediate runs

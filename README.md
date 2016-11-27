# cartoon-colorization

## Evaluation
There are two ways to evaluate a model on the sketch and reference dataset.

1. `evalPerReference.py` evaluates the full model on the sketch database and one reference image. Its usage is as follows:

   ```bash
   python evalPerReference.py --prototxt /path/to/deploy/ --caffemodel /path/to/caffemodel/
   ```

2. `evalPerSketch.py` evaluates the full model on one sketch image and the reference database. Its usage is as follows:

   ```bash
   python evalPerSketch.py --prototxt /path/to/deploy/ --caffemodel /path/to/caffemodel/
   ```
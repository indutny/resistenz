# reSistenZ

WIP

## How to export to CoreML

```sh
python3 python/save_graph.py saves/prior_sizes_iou0.7/00000600 saves/prior_sizes_iou0.7/
freeze_graph --input_graph ./saves/prior_sizes_iou0.7/graph.pbtxt --input_checkpoint=./saves/prior_sizes_iou0.7/graph.ckpt --output_node_names="resistenz/output/output" --output_graph=./pretrained/prior_sizes_iou0.7/freeze.pb
python3 python/convert.py pretrained/prior_sizes_iou0.7/freeze.pb pretrained/prior_sizes_iou0.7/graph.mlmodel
```

#### LICENSE

This software is licensed under the MIT License.

Copyright Fedor Indutny, 2018.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

---

Photos are downloaded from [flickr][0], and are subject to [creative commons
license][1].

[0]: https://github.com/indutny/resistenz/blob/master/dataset/urls.md
[1]: https://creativecommons.org/licenses/by/2.0/

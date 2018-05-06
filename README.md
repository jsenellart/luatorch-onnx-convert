# luatorch-onnx-convert

## Overview

This repository provides an utility to extract model(s) from a serialized lua-torch model (`.t7` file) and convert it/them to onnx format.  

## Requirements

* You need first to install lua torch as well as the library used in the model that are necessary to load the model from torch (for instance, `nn`, `nngraph`, `onmt` or your own libraries).
* Install protobuf lua library from this repository https://github.com/jsenellart/protobuf-lua (this version is mandatory since it fixes some issues with the original implementation).

You can regenerate the lua interpreter of the onnx proto file by doing:

```
protoc --lua_out=. onnx.proto
```

This will generate an updated version of `onnx_pb.lua`

## Extracting model

```
$ th convert.lua -t7 test/data/model1.t7 -require nn  -force
```

`convert.lua` goes through the serialized torch object and looks for supported modules or models. Each of them is converted into `test/data/model1.onnxdir` directory with the corresponding name:

```
model.linear-bias.onnx
model.linear-nobias.onnx
```

The option `-require nn` indicates which library are necessary to deserialize the model. GPU model needs `cunn` installation to be read.

## Convertors

To perform conversion to onnx, each of the used module must be described with onnx operators. New convertors for specific library can be added in `convertors`. Their name must match `onnx_class` where `class` is the torch classname of the modules.

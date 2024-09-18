# micrograd-cpp

C++ impl of [micrograd](https://github.com/karpathy/micrograd).

- YouTube: https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
- Building an DAG in C++, from operator overrides.
- A node can be shared by multiple children.
- Using smart-pointer to manage memory.

## Prerequisites

You need a C++ dev environment and CMake. I'm using Clang:

```
⋊> ~/C/m/debug on main ⨯ clang --version                                                                                                                                                                                                                          (base) 00:21:10
Apple clang version 15.0.0 (clang-1500.3.9.4)
Target: arm64-apple-darwin23.6.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

## Building

```
# Pull matplot++
git submodule init
git submodule update
# build
mkdir cmake-build-debug
cd cmake-build-debug && cmake -DCMAKE_BUILD_TYPE=Debug ..
make
./micrograd++
```

## Output

It runs the example in the Video.

```
Preds: [Value(data=0.149048), Value(data=0.853032), Value(data=0.550389), Value(data=0.596208)]
Loss: 6.7246
Preds: [Value(data=0.195786), Value(data=0.836784), Value(data=0.50433), Value(data=0.61548)]
Loss: 6.4314
Preds: [Value(data=0.224787), Value(data=0.815759), Value(data=0.443745), Value(data=0.623476)]
Loss: 6.1241
Preds: [Value(data=0.241228), Value(data=0.789126), Value(data=0.368526), Value(data=0.623298)]
Loss: 5.79148
Preds: [Value(data=0.250003), Value(data=0.756087), Value(data=0.280002), Value(data=0.617303)]
Loss: 5.4312
Preds: [Value(data=0.255546), Value(data=0.716055), Value(data=0.18155), Value(data=0.607584)]
Loss: 5.04911
Preds: [Value(data=0.261507), Value(data=0.668829), Value(data=0.0783758), Value(data=0.596168)]
Loss: 4.65633
...
Preds: [Value(data=0.913115), Value(data=-0.851749), Value(data=-0.907064), Value(data=0.894504)]
Loss: 0.0492936
Preds: [Value(data=0.913604), Value(data=-0.852617), Value(data=-0.90751), Value(data=0.895043)]
Loss: 0.0487563
Preds: [Value(data=0.914085), Value(data=-0.853472), Value(data=-0.907951), Value(data=0.895574)]
Loss: 0.0482298
```

## Replicating the demo

```
Number of parameters: 337
step 0 loss 0.867165, accuracy 56%
step 1 loss 0.348403, accuracy 85%
step 2 loss 0.303197, accuracy 85%
step 3 loss 0.267835, accuracy 88%
step 4 loss 0.261888, accuracy 88%
step 5 loss 0.257721, accuracy 88%
step 6 loss 0.253114, accuracy 88%
step 7 loss 0.247208, accuracy 88%
step 8 loss 0.242097, accuracy 90%
step 9 loss 0.239211, accuracy 90%
step 10 loss 0.236226, accuracy 90%
step 11 loss 0.23288, accuracy 90%
step 12 loss 0.229071, accuracy 89%
step 13 loss 0.224974, accuracy 90%
step 14 loss 0.220976, accuracy 91%
step 15 loss 0.217237, accuracy 91%
step 16 loss 0.213677, accuracy 91%
step 17 loss 0.210206, accuracy 91%
step 18 loss 0.206761, accuracy 92%
step 19 loss 0.203304, accuracy 92%
step 20 loss 0.199807, accuracy 92%
step 21 loss 0.196253, accuracy 92%
step 22 loss 0.192631, accuracy 93%
step 23 loss 0.188939, accuracy 93%
step 24 loss 0.18518, accuracy 93%
step 25 loss 0.181373, accuracy 93%
step 26 loss 0.177549, accuracy 93%
step 27 loss 0.173793, accuracy 94%
step 28 loss 0.170399, accuracy 93%
step 29 loss 0.168731, accuracy 95%
step 30 loss 0.174907, accuracy 93%
step 31 loss 0.180481, accuracy 93%
step 32 loss 0.196059, accuracy 91%
step 33 loss 0.157888, accuracy 94%
step 34 loss 0.15614, accuracy 94%
step 35 loss 0.162078, accuracy 94%
step 36 loss 0.175451, accuracy 93%
step 37 loss 0.1535, accuracy 95%
step 38 loss 0.159631, accuracy 93%
step 39 loss 0.149359, accuracy 94%
step 40 loss 0.150875, accuracy 94%
step 41 loss 0.138686, accuracy 96%
step 42 loss 0.135294, accuracy 95%
step 43 loss 0.126857, accuracy 96%
step 44 loss 0.122031, accuracy 96%
step 45 loss 0.11591, accuracy 96%
step 46 loss 0.109713, accuracy 96%
step 47 loss 0.10174, accuracy 97%
step 48 loss 0.0925694, accuracy 97%
step 49 loss 0.0841137, accuracy 99%
step 50 loss 0.0770053, accuracy 99%
step 51 loss 0.071882, accuracy 100%
step 52 loss 0.0704658, accuracy 99%
step 53 loss 0.0723406, accuracy 100%
step 54 loss 0.0915456, accuracy 97%
step 55 loss 0.0685904, accuracy 100%
step 56 loss 0.0833102, accuracy 97%
step 57 loss 0.0646067, accuracy 100%
step 58 loss 0.0802135, accuracy 97%
step 59 loss 0.0553207, accuracy 100%
step 60 loss 0.0615925, accuracy 99%
step 61 loss 0.051684, accuracy 100%
step 62 loss 0.0574414, accuracy 99%
step 63 loss 0.0467224, accuracy 100%
step 64 loss 0.0482662, accuracy 99%
step 65 loss 0.0419762, accuracy 100%
step 66 loss 0.0413639, accuracy 100%
step 67 loss 0.0379018, accuracy 100%
step 68 loss 0.0366942, accuracy 100%
step 69 loss 0.0350696, accuracy 100%
step 70 loss 0.0341801, accuracy 100%
step 71 loss 0.0334082, accuracy 100%
step 72 loss 0.0328326, accuracy 100%
step 73 loss 0.0323332, accuracy 100%
step 74 loss 0.0318875, accuracy 100%
step 75 loss 0.0314743, accuracy 100%
step 76 loss 0.0310859, accuracy 100%
step 77 loss 0.0307183, accuracy 100%
step 78 loss 0.0303693, accuracy 100%
step 79 loss 0.0300373, accuracy 100%
step 80 loss 0.0297211, accuracy 100%
step 81 loss 0.0294195, accuracy 100%
step 82 loss 0.0291316, accuracy 100%
step 83 loss 0.0288564, accuracy 100%
step 84 loss 0.0285932, accuracy 100%
step 85 loss 0.0283411, accuracy 100%
step 86 loss 0.0280996, accuracy 100%
step 87 loss 0.0278679, accuracy 100%
step 88 loss 0.0276455, accuracy 100%
step 89 loss 0.0274319, accuracy 100%
step 90 loss 0.0272266, accuracy 100%
step 91 loss 0.0270291, accuracy 100%
step 92 loss 0.026839, accuracy 100%
step 93 loss 0.0266559, accuracy 100%
step 94 loss 0.0264795, accuracy 100%
step 95 loss 0.0263094, accuracy 100%
step 96 loss 0.0261452, accuracy 100%
step 97 loss 0.0259868, accuracy 100%
step 98 loss 0.0258338, accuracy 100%
step 99 loss 0.025686, accuracy 100%
```

<img src="./demo.png" width="560" alt="demo input">

<p>Decision boundary</p>

<img src="./decision-boundary.png" width="560" alt="decision boundary">

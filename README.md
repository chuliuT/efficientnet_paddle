# efficientnet_paddle

borrow some code (Conv2dStaticSamePadding) from ![](https://github.com/GuoQuanhao/EfficientDet-Paddle/efficientnet)

apply trans func  all b0-b7 align pytorch  at 1e-5

```
import numpy as np
import torch

paddle.set_device("cpu")
device = torch.device("cpu")
torch.set_printoptions(precision=8)

from efficientnet_pytorch import EfficientNet as Eff_torch


def EfficientNet_trans_paddle(model_name='efficientnet-b0', shape=[1, 3, 240, 240], prec=1e-5,
                              save_dir="../paddle_ckpt"):

    inputs = np.random.rand(*shape).astype("float32")
    model = EfficientNet.from_pretrained(model_name)

    model2 = Eff_torch.from_pretrained(model_name)
    model2.eval()
    new_st = {}
    st = model2.state_dict()
    for k in st.keys():
        if "num_batches_tracked" in k:
            continue
        new_st[k] = st[k]

    for ((name, param), key_t) in zip(model.named_parameters(), new_st.keys()):
        print(name, param.shape, param.dtype)
        print(key_t, new_st[key_t].shape, new_st[key_t].dtype)

        if "_fc.weight" in name:
            param.set_value(new_st[key_t].numpy().astype("float32").transpose((1, 0)))
        else:
            param.set_value(new_st[key_t].numpy().astype("float32"))
        print("**" * 10)
    print("model weight trans done...")
    model.eval()
    model2.eval()

    inputs1 = paddle.to_tensor(inputs)
    inputs2 = torch.Tensor(inputs).to(device)

    out = model(inputs1)
    out2 = model2(inputs2)
    print(out[0, :10])
    # print(out2[0, :10])
    print(out2[0, :10])
    print(np.allclose(out.cpu().numpy(), out2.data.cpu().numpy(), atol=prec))
    assert np.allclose(out.cpu().numpy(), out2.data.cpu().numpy(), atol=prec)
    paddle.save(model.state_dict(), f"{save_dir}/{model_name}.pdparams")


'''
Model           |  input_size  |  width_coefficient  |  depth_coefficient  | dropout_rate
-------------------------------------------------------------------------------------------
EfficientNetB0  |   224x224    |    1.0              |      1.0            |    0.2
-------------------------------------------------------------------------------------------
EfficientNetB1  |   240x240    |    1.0              |      1.1            |    0.2
-------------------------------------------------------------------------------------------
EfficientNetB2  |   260x260    |    1.1              |      1.2            |    0.3
-------------------------------------------------------------------------------------------
EfficientNetB3  |   300x300    |    1.2              |      1.4            |    0.3
-------------------------------------------------------------------------------------------
EfficientNetB4  |   380x380    |    1.4              |      1.8            |    0.4
-------------------------------------------------------------------------------------------
EfficientNetB5  |   456x456    |    1.6              |      2.2            |    0.4
-------------------------------------------------------------------------------------------
EfficientNetB6  |   528x528    |    1.8              |      2.6            |    0.5
-------------------------------------------------------------------------------------------
EfficientNetB7  |   600x600    |    2.0              |      3.1            |    0.5
'''

EfficientNet_trans_paddle("efficientnet-b0", shape=[1, 3, 224, 224])
EfficientNet_trans_paddle("efficientnet-b1", shape=[1, 3, 240, 240])
EfficientNet_trans_paddle("efficientnet-b2", shape=[1, 3, 260, 260])
EfficientNet_trans_paddle("efficientnet-b3", shape=[1, 3, 300, 300])
EfficientNet_trans_paddle("efficientnet-b4", shape=[1, 3, 380, 380])
EfficientNet_trans_paddle("efficientnet-b5", shape=[1, 3, 456, 456])
EfficientNet_trans_paddle("efficientnet-b6", shape=[1, 3, 528, 528])
EfficientNet_trans_paddle("efficientnet-b7", shape=[1, 3, 600, 600])
```

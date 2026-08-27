
#python modelsCollect9WIED99Lost.py --batch_size 750 --test_size 0.85 --epochs 300 --lr 0.0002 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1

#python modelsCollect9W99NolgLost.py --batch_size 1100 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1#训练vanishTime
#python modelsCollect9W99NolgLost.py --batch_size 1100 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1

#没有丢失数据状态下，mlp_cf跟车模型0的训练
python model9w99RTlost.py --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1
python model9w99RTlost.py --batch_size 1100 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 0 --fixdata 0 --trainvalmode 0 --goffset 1

#没有丢失数据状态下，mlp_reg回归模型1的训练
python model9w99RTlost.py --batch_size 300 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 2000 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1
python model9w99RTlost.py --batch_size 1100 --test_size 0.85 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 0 --model 1 --fixdata 0 --trainvalmode 0 --goffset 1

#有有丢失数据状态下，丢失slot的预测模型回归模型2的训练
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 2 --fixdata 0 --trainvalmode 1 --goffset 1

#有有丢失数据状态下，采用已经训练好的模型，先slot预测，*再修补*，最后cf的vanishTime模型3的验证
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 3 --fixdata 3 --trainvalmode 1 --goffset 1


#有有丢失数据状态下，采用已经训练好的模型，先slot预测，*再修补*，最后cf的vanishTime模型3的验证
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 4 --fixdata 3 --trainvalmode 1 --goffset 1

#有有丢失数据状态下，采用已经训练好的模型，先slot预测，*不修补*，最后cf的vanishTime模型3的验证
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 3 --fixdata 0 --trainvalmode 1 --goffset 1

#有有丢失数据状态下，采用已经训练好的模型，先slot预测，*不修补*，最后回归的vanishTime模型3的验证
python model9w99RTlost.py --batch_size 2500 --test_size 0.5 --epochs 1000 --lr 0.0001 --unit 256 --layNum 128 --dt 0.1 --nC 5000 --model 4 --fixdata 0 --trainvalmode 1 --goffset 1

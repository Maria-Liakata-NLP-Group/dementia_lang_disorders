import argparse


def parse_arguments(): 
     parser = argparse.ArgumentParser()
     
    
     parser.add_argument('--num_classes', default=4, type=int, help='num of classes')
     parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
     parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train for')
     parser.add_argument('--checkpoint_path', default="tmp", help='path to save the checkpoint model')
     parser.add_argument('--model_name', default="RoBERTa_Prompt_dem", choices=['RoBERTa', 'RoBERTa_MLM', 'RoBERTa_CLS', 'RoBERTa_Multitask', 'RoBERTa_Prompt', 'RoBERTa_Prompt_dem', 'RoBERTa_entail', 'RoBERTa_Prompt_inverse'], help='model name')
     parser.add_argument('--task', default="fine-tuning", choices=['fine-tuning'], help='type of task')
     
     
     parser.add_argument('--batch_size', type=int, default=32, help='batch size')
     parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
     parser.add_argument('--max_grad_norm', type=float, default=1.0, help= 'Gradient clip')
     parser.add_argument('--patience', type=int, default=4, help= 'Early stop ')
     parser.add_argument('--alpha', type=int, default=0.25, help='focal loss alpha')
     parser.add_argument('--gamma', type=int, default=2, help='focal loss gamma')
     parser.add_argument('--statistic_step', type=int, default=20, help='show statistics per a number of step')
          
     opt = parser.parse_args()
     return opt

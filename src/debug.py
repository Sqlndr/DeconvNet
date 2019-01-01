from optparse import OptionParser

parser = OptionParser()


parser.add_option("-m", "--model",
                  dest="model",
                  type=str,
                  help="model's name",
                  default="nodel1")



parser.add_option("", "--method",
                  dest="method",
                  type=str,
                  help="optimizer",
                  default="Adam")




parser.add_option("", "--metric",
                  dest="metric",
                  type=str,
                  help="optimizer",
                  default="Adam")



parser.add_option("", "--lr",
                  dest="lr",
                  type=float,
                  help="model's Learning Rate",
                  default=0.03)


parser.add_option("", "--decay",
                  dest="decay",
                  type=float,
                  help="model's Decay Rate for Adam Optimizer",
                  default=0.03)


parser.add_option("", "--beta_1",
                  dest="deta_1",
                  type=float,
                  help="model's Beta 1 Rate for Adam Optimizer",
                  default=0.9)


parser.add_option("", "--beta_2",
                  dest="deta_2",
                  type=float,
                  help="model's Beta 2 Rate for Adam Optimizer",
                  default=0.99)



if __name__=="__main__":

    print("start debug")

    (args, _) = parser.parse_args()

    print(args)
    print(args.model)




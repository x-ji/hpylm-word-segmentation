# __precompile__()

# module HPYLMTrain

using ArgParse
push!(LOAD_PATH, "./")
import HPYLM

function main(args)
    s = ArgParseSettings(description = "Train n-gram model")
    @add_arg_table s begin
        "--corpus", "-c"
            help="training corpus"
            required=true
        "--order", "-o"
            help="order of the model"
            arg_type = Int
            required=true
        "--iter", "-i"
            help="number of iterations for the model"
            arg_type = Int
            required = true
        "--output", "-m"
            help="model output path"
            required = true
    end

    parsed = parse_args(args, s)

    HPYLM.train(parsed["corpus"], parsed["order"], parsed["iter"], parsed["output"])
end

# end

main(ARGS)

# push!(LOAD_PATH, "./")
# import HPYLMTrain

# HPYLMTrain.main(ARGS)

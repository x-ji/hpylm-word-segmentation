using ArgParse

push!(LOAD_PATH, "./")
import HPYLM

function main(args)
    s = ArgParseSettings(description = "Evaluate n-gram model")
    @add_arg_table s begin
        "--corpus", "-c"
            help="evaluation corpus"
            required=true
        "--model", "-m"
            help="previously trained model"
            required=true
    end

    parsed = parse_args(args, s)

    HPYLM.evaluate(parsed["corpus"], parsed["model"])
end

main(ARGS)

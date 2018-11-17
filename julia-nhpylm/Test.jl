mutable struct A
    x::Int
    y::Union{Nothing, Int}
    function A()
        a = new()
        a.x = 5
        # Note that if `Nothing` is a possible value, you simply don't need to do anything. Trying to access it won't throw any error!
        return a
    end
end
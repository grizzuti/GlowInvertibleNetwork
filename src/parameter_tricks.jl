Base.:*(α::Number, p::Parameter) = Parameter(α*p.data)
Base.:*(p::Parameter, α::Number) = α*p
Base.:*(α::Number, p::AbstractArray{Parameter,1}) = α.*p
Base.:*(p::AbstractArray{Parameter,1}, α::Number) = p*α
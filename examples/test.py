def function(arg_1, arg_2, args, **kwargs):
    print(arg_1)
    print(arg_2)
    print(args)
    print(kwargs)
    # for arg in args:
    #     print(arg)
    # for key, value in kwargs.items():
    #     print(key, value)
function(1, 2, 3, 4, 5, a=6, b=7)

#!/usr/bin/env python
# _*_ coding: utf-8 _*_
def time_analyze(func):
    """
    此函数用来计算函数运行时间，做装饰器
    :param func:
    :return:
    """
    from time import perf_counter
    # exec_times = 1

    def callf(*args, **kwargs):
        start = perf_counter()
        # for i in range(exec_times):
        r = func(*args, **kwargs)
        finish = perf_counter()
        print("{:<20}{:10.6} s".format(func.__name__ + ":", finish - start))
        return r
    return callf


@time_analyze
def _slowest_replace_(original_str):
    replace_list = []
    for i, char in enumerate(original_str):
        c = char if char != " " else "-"
        replace_list.append(c)
    return "".join(replace_list)


@time_analyze
def _slow_replace_(original_str):
    replace_str = ""
    for i, char in enumerate(original_str):
        c = char if char != " " else "-"
        replace_str += c
    return replace_str


@time_analyze
def _fast_replace_(original_str):
    return "-".join(original_str.split())


@time_analyze
def _fastest_replace_(original_str):
    return original_str.replace(" ", "-")


if __name__ == "__main__":
    original_str = "People's Republic of China"

    res1 = _slowest_replace_(original_str)
    res2 = _slow_replace_(original_str)
    res3 = _fast_replace_(original_str)
    res4 = _fastest_replace_(original_str)
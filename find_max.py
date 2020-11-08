import sys

if __name__ == "__main__":
    pth = sys.argv[1]
    macro_list, micro_list = [], []
    max_macro, max_micro = None, None
    with open(f'{pth}/test_macro_f1', 'r') as f_macro, open(f'{pth}/test_micro_f1', 'r') as f_micro:
        for row in f_macro.readlines():
            macro_list.append(round(float(row), 3))
        for row in f_micro.readlines():
            micro_list.append(round(float(row), 3))
        
    for macro, micro in zip(macro_list, micro_list):
        if max_macro is None:
            max_macro, max_micro = macro, micro
        elif macro == max_macro:
            if micro > max_micro:
                max_micro = micro
        elif macro > max_macro:
            if micro >= max_micro:
                max_macro = macro
                max_micro = micro
            else:
                if macro - max_macro > max_micro - micro:
                    max_macro = macro
                    max_micro = micro
    print(f'macro: {max_macro}, micro: {max_micro}')
import inspect
from operator import index
import re

name_length = 50
result_length = 30


def print_line(i, result=None) -> str:
    line = re.split("(\s+)", i)
    name = line[1] + " ".join(i.split()).replace('"', '\\"')
    if len(name) >= name_length:
        name = name[: name_length - 5] + " ... "

    if result == None:
        out = line[1] + f'print(f"{name[:name_length]:<{name_length}}")\n'
    else:
        out = line[1] + f'print(f"{name[:name_length]:<{name_length}}-> {{\'\'.join(str({result}).split())[:{result_length}]}}")\n'
    return out


def print_all():
    def func(i) -> list:
        outstr = []
        outstr.append(i)
        outstr.append(print_line(i))
        return outstr

    return func


def print_custom(string, pre: bool = False, additional_indent=0):
    def func(i) -> list:
        outstr = []
        line = re.split("(\s+)", i)
        if pre:
            outstr.append(i)
        outstr.append(
            line[1] + " " * (4 * additional_indent) + f"print(\"{line[1] + ' '*(4*additional_indent)}{string}\")\n"
        )
        if not pre:
            outstr.append(i)
        return outstr

    return func


def print_left_of(string):
    def func(i: str) -> list:
        outstr = []
        outstr.append(i)
        pos = i.find(string)
        if "," in i[:pos]:
            outstr.append(print_line(i))
            for val in re.split("(?:,)",i[:pos]):
                outstr.append(print_line(i + ":" + val,val))
        else:
            outstr.append(print_line(i, i[:pos]))
        return outstr

    return func


def print_between(start: str, stop: str, prefix=None, pre: bool = False):
    def func(i: str) -> list:
        outstr = []
        pos1 = i.find(start)
        pos2 = i.rfind(stop)
        if pre:
            outstr.append(i)
        if prefix == None:
            outstr.append(print_line(i, i[pos1 + len(start) : pos2]))
        else:
            outstr.append(print_line(i, prefix + i[pos1 + len(start) : pos2]))
        if not pre:
            outstr.append(i)
        return outstr

    return func


class log_class:
    index = 0
    log_level = 0

    def __init__(self) -> None:
        pass

    def log(self, rules={}):
        def logger(fnc):
            code, cline = inspect.getsourcelines(fnc)
            fnc_name = str()
            outstr = []
            indent_pos = 0
            file = inspect.getfile(fnc)
            # print(inspect.getmembers(fnc))
            for ip in range(len(code)):
                i = code[ip]
                i = i[indent_pos:]
                found_in_rules = False
                if i.lstrip()[0] == "#":
                    continue
                if "@logger" in i:
                    indent_pos = i.find("@")
                    found_in_rules = True
                if "def" in i:
                    found_in_rules = True
                    pos = i.find("(")
                    pos2 = i.rfind(")")
                    outstr.append(i[:pos] + str(self.index) + i[pos:])
                    fnc_name = fnc.__name__ + str(self.index)
                    self.index += 1
                    outstr.append('    print("FUNCTON:' + fnc.__qualname__ + ":" + f"{cline+ip}" + f"\t{file}" + '")\n')

                    super_exists = False
                    for t in range(len(code)):
                        if "super" in code[t]:
                            super_exists = True
                            break

                    if not super_exists:
                        sig = inspect.signature(fnc)
                        # print(sig)
                        for s in sig.parameters:
                            name = "    " + str(sig.parameters[s])
                            outstr.append(f'    print(f"{name:<{name_length}}-> {{str({s})[:{result_length}]}}")\n')
                        outstr.append(f'    print(f"  CONTENT:")\n')
                        found_in_rules = True
                    continue
                if "super" in i:
                    outstr.append(i)
                    sig = inspect.signature(fnc)
                    # print(sig)
                    for s in sig.parameters:
                        name = "    " + str(sig.parameters[s])
                        outstr.append(f'    print(f"{name:<{name_length-2}}-> {{str({s})[:{result_length}]}}")\n')
                    outstr.append(f'    print(f"  CONTENT:")\n')
                    found_in_rules = True
                    continue
                for case in rules:
                    if case in i:
                        outstr.extend(rules[case](i))
                        found_in_rules = True
                        break
                if not found_in_rules:
                    outstr.append(i)
            # print(''.join(outstr))
            exec("".join(outstr), fnc.__globals__)
            return eval(fnc_name, fnc.__globals__)

        return logger


logger = log_class()

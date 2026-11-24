"""Measure tracked source complexity and summarize coverage JSON from the audit runs."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from radon.visitors import ComplexityVisitor

REPO=Path(__file__).resolve().parents[2]
ROOTS=("src","tools","experiments","results","dataset/tools")
FIELDS=("covered_lines","num_statements","covered_branches","num_branches")


def tracked(*roots):
    result=subprocess.check_output(["git","ls-files","-z","--",*roots],cwd=REPO)
    return [p for p in result.decode().split("\x00") if p.endswith(".py") and not p.startswith("tools/criteria/")]


def complexity():
    scores=[]
    for path in tracked("src","tools"):
        tree=ast.parse((REPO/path).read_text(encoding="utf-8-sig"))
        for fn in ast.walk(tree):
            if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):
                node=ast.Module(body=[fn],type_ignores=[])
                value=ComplexityVisitor.from_ast(node).functions[0].complexity
                scores.append({"file":path,"line":fn.lineno,"name":fn.name,"cc":value})
    counts=Counter("<=10" if x["cc"]<=10 else "11-20" if x["cc"]<=20 else ">=21" for x in scores)
    return {"total":len(scores),"counts":dict(counts),"top":sorted(scores,key=lambda x:-x["cc"])[:12]}


def coverage_summary(file):
    payload=json.loads(file.read_text(encoding="utf-8-sig"))
    per_path={k.replace("\\","/"):v["summary"] for k,v in payload["files"].items()}
    sums={}
    for name,folders in (("all_tracked",ROOTS),("src_tools",("src","tools")),("src",("src",)),("predict",())):
        paths=[p for p in per_path if any(p.startswith(x+"/") for x in folders)] if folders else ["src/predict.py"]
        group={k:sum(per_path[p][k] for p in paths if p in per_path) for k in FIELDS}
        sums[name]={"files":len(paths),**group}
    return {"coverage_version":payload["meta"]["version"],"scopes":sums}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--coverage",type=Path,nargs="*")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result={"complexity":complexity()}
    result["coverage"]={path.stem:coverage_summary(path) for path in a.coverage or []}
    a.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")

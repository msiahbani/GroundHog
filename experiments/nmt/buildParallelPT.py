import sys

if __name__ == "__main__":
    fi = open(sys.argv[1], "r")
    fo1 = open(sys.argv[2]+".par1", "w")
    fo2 = open(sys.argv[2]+".par2", "w")
    for l in fi:
	(src, tgt) = l.strip().split(" ||| ")[0:2]
	print>>fo1, src
	print>>fo2, tgt
    fo1.close
    fo2.close

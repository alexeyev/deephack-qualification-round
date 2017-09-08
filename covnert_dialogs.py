# coding:utf-8

import re

filename = "data/validation_upd.txt"
pattern = re.compile("(<\w+_speaker>)")
id = 0

firstline = True

with open("validation_upd_spl.txt", "w+") as w:

    for line in open(filename, "r+"):

        if firstline:
            firstline = False
            continue

        splitted = line.strip().split("\t")

        context = splitted[id].replace("|", "*")

        before = splitted[:id]
        after = splitted[id + 1:]
        creplies = pattern.split(context)[1:]

        # print(creplies)

        reunited = [creplies[i] + creplies[i + 1] for i in range(0, len(creplies), 2)]

        # print(reunited)

        w.write(("\t".join(before).strip() + "\t" + "\t".join(after).strip() + "\t" + "|".join(reunited)).strip() + "\n")

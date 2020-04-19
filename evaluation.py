from matplotlib import pyplot as ppt
from statistics import mean

def precision_recall(res_set, ide_rel):
    p_r = {}
    hits = 0
    rss_len = len(res_set)
    idr_len = len(ide_rel)
    for i, res in enumerate(res_set):
        if res in ide_rel:
            hits += 1
            recall = hits/idr_len
            precision = hits/(i+1)
            p_r[recall] = precision
    return p_r, hits/idr_len, hits/rss_len

def multi_p_r(queries_file, ideal_file, top_10k):
    ide_rel = []
    with open(ideal_file, "r", encoding="utf-8") as ifr:
        for line in ifr.readlines():
            ide_rel.append(line.strip())
    res_sets = {}
    with open(queries_file, "r", encoding="utf-8") as qsfr:
        for line in qsfr.readlines():
            line = line.strip()
            file = line.replace(" ", "+")
            res_sets[line] = []
            with open("query_results/{}_top_{}.txt".format(file, top_10k*10), "r", encoding="utf-8") as qfr:
                for l in qfr.readlines():
                    l = l.strip()
                    res_sets[line].append(l)
    with open("evaluation_results.txt", "w", encoding="utf-8") as w:
        prs = []
        ps = []
        avg_pr = {}
        for query, rs in res_sets.items():
            p_r, r, p = precision_recall(rs, ide_rel)
            prs.append(p_r)
            ps.append(p)
            w.write("{}:   {}, recall: {} precision: {}\n".format(query,p_r, r, p))
            x, y = zip(*(sorted(p_r.items())))
            ppt.ylabel('Precision')
            ppt.xlabel('Recall')
            ppt.title("'{}' Precision-Recall Curve".format(query))
            ppt.plot(x, y, linestyle='--', marker='o')
            ppt.savefig("{}_p_r.png".format(query))
            ppt.close()
        w.write("\nAverage precision: {}".format((mean(ps))))
        for p_r in prs:
            for r in p_r.keys():
                if r in avg_pr.keys():
                    avg_pr[r].append(p_r[r])
                else:
                    avg_pr[r] = [p_r[r]]
        res = {}
        for r, ps in avg_pr.items():
            res[r] = mean(ps)
        w.write("\nAverage Precision-recall:\n")
        w.writelines("Recall:{}, Precision:{}\n".format(r,p) for r, p in sorted(res.items()))
        x, y = zip(*(sorted(res.items())))
        ppt.ylabel('Precision')
        ppt.xlabel('Recall')
        ppt.title("Average Precision-Recall Curve")
        ppt.plot(x, y, linestyle='--', marker='o')
        ppt.savefig("avg_pr.png".format(query))
        ppt.close()



if __name__ == '__main__':
    multi_p_r("queries.txt", "ideal_docs.txt", 10)

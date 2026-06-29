import io
p = "/pscratch/sd/j/joe2k/zr/zr_eval.py"
s = open(p).read()

a_old = '    p.add_argument("--holdout-frac", type=float, default=0.05)'
a_new = a_old + '\n    p.add_argument("--train-split", action="store_true", help="use the 95%% train side of the healpix split")'
assert a_old in s and a_new not in s, "arg patch precondition failed"
s = s.replace(a_old, a_new)

b_old = (
    "        if args.full_corpus:\n"
    "            use = records\n"
    "        else:\n"
    "            _, use = split_records_by_healpix(records, holdout_frac=args.holdout_frac, seed=42)\n"
)
b_new = (
    "        if args.full_corpus:\n"
    "            use = records\n"
    "        elif args.train_split:\n"
    "            use, _ = split_records_by_healpix(records, holdout_frac=args.holdout_frac, seed=42)\n"
    "        else:\n"
    "            _, use = split_records_by_healpix(records, holdout_frac=args.holdout_frac, seed=42)\n"
)
assert b_old in s, "split patch precondition failed"
s = s.replace(b_old, b_new)
open(p, "w").write(s)
print("patched OK")

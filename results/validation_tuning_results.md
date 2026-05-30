# Explicit ALS Validation Tuning Results

This file records one explicit-feedback ALS validation tuning run.

| rank | regParam | maxIter | NDCG@100 | Recall@100 | MAP | Eval users |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.05 | 5 | 0.0000781876 | 0.0000892388 | 0.0000058769 | 4,936 |
| 5 | 0.1 | 5 | 0.0000659120 | 0.0000858286 | 0.0000044108 | 328 |
| 5 | 0.5 | 5 | 0.0000697059 | 0.0001270981 | 0.0000044673 | 4,936 |
| 10 | 0.05 | 5 | 0.0002064832 | 0.0003265090 | 0.0000123828 | 4,936 |
| 10 | 0.1 | 5 | 0.0001302299 | 0.0002208884 | 0.0000066328 | 4,936 |
| 10 | 0.5 | 5 | 0.0002530373 | 0.0003986629 | 0.0000396273 | 4,936 |
| 20 | 0.05 | 5 | 0.0004976388 | 0.0008312581 | 0.0000475074 | 4,936 |
| 20 | 0.1 | 5 | 0.0004034873 | 0.0006981561 | 0.0000294104 | 4,935 |
| 20 | 0.5 | 5 | 0.0005677491 | 0.0009798163 | 0.0000593816 | 4,935 |

## Best Validation Setting

```text
rank = 20
regParam = 0.5
maxIter = 5
NDCG@100 = 0.0005677491
Recall@100 = 0.0009798163
MAP = 0.0000593816
num_eval_users = 4935
```

## Note

These explicit-only results should be interpreted as a rating-based ALS baseline. In the broader project, implicit-feedback and combined-feedback ALS performed substantially better because they used more complete behavioral signals from the Goodreads interaction data.

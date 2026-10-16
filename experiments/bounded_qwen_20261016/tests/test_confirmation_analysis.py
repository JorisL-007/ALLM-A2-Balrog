import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('confirmation_analysis',Path(__file__).resolve().parents[1]/'protocol/analysis.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


class ConfirmationTests(unittest.TestCase):
    def test_benchmarks_have_equal_weight(self):
        rows=[]
        for env,count,value in [('babyai',5,1),('crafter',1,0),('babaisai',6,0)]:
            rows += [dict(seed=11,variant='actor',pass_id='one',environment=env,progression=value) for _ in range(count)]
        self.assertAlmostEqual(analysis.seed_scores(rows)[11,'actor'],1/3)

    def test_repeating_layouts_does_not_add_independent_clusters(self):
        rows=[]
        for seed,value in [(11,0),(12,1)]:
            for env in ['babyai','crafter','babaisai']:
                rows.append(dict(seed=seed,variant='actor',pass_id='one',environment=env,progression=value))
        single=analysis.seed_scores(rows)
        repeated=analysis.seed_scores(rows+[dict(r,pass_id='two') for r in rows])
        self.assertEqual(single,repeated)
        ci=analysis.paired_cluster_interval(list(repeated.values()),draws=1000)
        self.assertEqual(ci['seed_clusters'],2)

    def test_missing_benchmark_cannot_silently_be_reweighted(self):
        with self.assertRaisesRegex(ValueError,'Missing benchmark'):
            analysis.seed_scores([dict(seed=11,variant='actor',pass_id='one',environment='babyai',progression=1)])


if __name__=='__main__':unittest.main()

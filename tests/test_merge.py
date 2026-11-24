
import pandas as pd
def test_merge_by_id():
    a=pd.DataFrame({"candidate_id":["A","B"],"x":[1,2]})
    b=pd.DataFrame({"candidate_id":["B","A"],"y":[20,10]})
    m=a.merge(b,on="candidate_id",validate="one_to_one")
    assert m.set_index("candidate_id").loc["A","y"]==10

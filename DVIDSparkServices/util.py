def select_item(rdd, *indexes):
    """
    Given an RDD of tuples, return an RDD listing the Nth item from each tuple.
    If the tuples are nested, you can provide multiple indexes to drill down to the element you want.
    
    >>> rdd = sc.parallelize([('foo', ('a', 'b')), ('bar', ('c', 'd'))])
    >>> select_item(rdd, 1, 0).collect()
    ['b', 'd']
    """
    def _select(item):
        for i in indexes:
            item = item[i]
        return item
    return rdd.map(_select)

def zip_many(*rdds):
    """
    Like RDD.zip(), but supports more than two RDDs.
    It's baffling that PySpark doesn't include this by default...
    """
    assert len(rdds) >= 2

    result = rdds[0].zip(rdds[1])
    rdds = rdds[2:]

    def append_value_to_key(k_v):
        return (k_v[0] + (k_v[1],))

    while rdds:
        next_rdd, rdds = rdds[0], rdds[1:]
        result = result.zip(next_rdd).map(append_value_to_key)
    return result

def join_many(*rdds):
    """
    Like RDD.join(), but supports more than two RDDs.
    It's baffling that PySpark doesn't include this by default...
    """
    assert len(rdds) >= 2
    
    result = rdds[0].join(rdds[1])
    rdds = rdds[2:]
    
    def condense_value(k_v):
        k, (v1, v2) = k_v
        return (k, v1 + (v2,))
    
    while rdds:
        next_rdd, rdds = rdds[0], rdds[1:]
        result = result.join(next_rdd).map(condense_value, True)
    return result

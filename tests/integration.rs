mod range_num_map;
use range_num_map::RangeNumMap;

mod test_range_num_map {
    use super::*;

    #[test]
    fn basic() {
        let mut range_map = RangeNumMap::new();
        range_map.insert(1..10, 0);
        assert_eq!(range_map.get(0), None);
        assert_eq!(range_map.get(7), Some(0));
        assert_eq!(range_map.get(8), Some(0));
        assert_eq!(range_map.get(9), Some(0));
        assert_eq!(range_map.get(10), None);
        range_map.insert(2..8, 1);
        assert_eq!(range_map.get(1), Some(0));
        assert_eq!(range_map.get(2), Some(1));
        assert_eq!(range_map.get(7), Some(1));
        assert_eq!(range_map.get(8), Some(0));
        assert_eq!(range_map.get(9), Some(0));
        assert_eq!(range_map.get(10), None);
    }
}

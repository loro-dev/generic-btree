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

    #[test]
    fn write_buffer_get_item() {
        let mut range_map = RangeNumMap::new();
        for i in 0..100 {
            range_map.insert(i..i + 1, i as isize);
        }
        for i in 0..100 {
            assert_eq!(range_map.get(i), Some(i as isize))
        }

        assert_eq!(range_map.get(100), None);
        range_map.insert(0..100, 100);
        assert_eq!(range_map.get(100), None);
        for i in 0..100 {
            assert_eq!(range_map.get(i), Some(100))
        }
    }

    #[test]
    fn write_buffer_iter() {
        let mut range_map = RangeNumMap::new();
        for i in 0..100 {
            range_map.insert(i..i + 1, i as isize);
        }

        range_map.insert(50..100, 100);
        for (r, value) in range_map.iter() {
            if r.start < 50 {
                assert_eq!(value, r.start as isize);
            } else {
                assert_eq!(value, 100);
            }
        }
    }

    #[test]
    fn write_buffer_drain() {
        let mut range_map = RangeNumMap::new();
        for i in 0..100 {
            range_map.insert(i..i + 1, i as isize);
        }

        range_map.insert(50..80, 100);
        for (r, value) in range_map.drain(25..100) {
            if r.start < 50 || r.start >= 80 {
                assert_eq!(value, r.start as isize);
            } else {
                assert_eq!(value, 100);
            }
        }
    }
}

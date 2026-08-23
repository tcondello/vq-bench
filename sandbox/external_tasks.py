"""
Terminal Bench Sandbox External Repository Tasks (N=100).
50 tasks from tokio-rs/tokio (Rust) and 50 tasks from tiangolo/fastapi (Python).
"""

EXTERNAL_TASKS = [
    {
        "id": "tokio_01",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Pending struct in Tokio and how it handles stream for the [`pending`](fn@pending) function..",
        "target_file": "tokio/tokio-stream/src/pending.rs",
        "target_symbols": [
            "Pending"
        ],
        "expected_snippet_keywords": [
            "Pending",
            "struct"
        ]
    },
    {
        "id": "tokio_02",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Empty struct in Tokio and how it handles stream for the [`empty`](fn@empty) function..",
        "target_file": "tokio/tokio-stream/src/empty.rs",
        "target_symbols": [
            "Empty"
        ],
        "expected_snippet_keywords": [
            "Empty",
            "struct"
        ]
    },
    {
        "id": "tokio_03",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Iter struct in Tokio and how it handles stream for the [`iter`](fn@iter) function..",
        "target_file": "tokio/tokio-stream/src/iter.rs",
        "target_symbols": [
            "Iter"
        ],
        "expected_snippet_keywords": [
            "Iter",
            "struct"
        ]
    },
    {
        "id": "tokio_04",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the new fn in Tokio and how it handles create a new `streamnotifyclose`..",
        "target_file": "tokio/tokio-stream/src/stream_close.rs",
        "target_symbols": [
            "new"
        ],
        "expected_snippet_keywords": [
            "new",
            "fn"
        ]
    },
    {
        "id": "tokio_05",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the into_inner fn in Tokio and how it handles /// returns `none` if the stream has reached its end..",
        "target_file": "tokio/tokio-stream/src/stream_close.rs",
        "target_symbols": [
            "into_inner"
        ],
        "expected_snippet_keywords": [
            "into_inner",
            "fn"
        ]
    },
    {
        "id": "tokio_06",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the StreamExt trait in Tokio and how it handles [futures-streamext]: https://docs.rs/futures/0.3/futures/stream/trait.streamext.html.",
        "target_file": "tokio/tokio-stream/src/stream_ext.rs",
        "target_symbols": [
            "StreamExt"
        ],
        "expected_snippet_keywords": [
            "StreamExt",
            "trait"
        ]
    },
    {
        "id": "tokio_07",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the merge_size_hints fn in Tokio and how it handles merge the size hints from two streams..",
        "target_file": "tokio/tokio-stream/src/stream_ext.rs",
        "target_symbols": [
            "merge_size_hints"
        ],
        "expected_snippet_keywords": [
            "merge_size_hints",
            "fn"
        ]
    },
    {
        "id": "tokio_08",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Once struct in Tokio and how it handles stream for the [`once`](fn@once) function..",
        "target_file": "tokio/tokio-stream/src/once.rs",
        "target_symbols": [
            "Once"
        ],
        "expected_snippet_keywords": [
            "Once",
            "struct"
        ]
    },
    {
        "id": "tokio_09",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the poll_next_entry fn in Tokio and how it handles polls the next value, includes the vec entry index.",
        "target_file": "tokio/tokio-stream/src/stream_map.rs",
        "target_symbols": [
            "poll_next_entry"
        ],
        "expected_snippet_keywords": [
            "poll_next_entry",
            "fn"
        ]
    },
    {
        "id": "tokio_10",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the poll_next_many fn in Tokio and how it handles zero only if the `streammap` is empty (or if `limit` is zero)..",
        "target_file": "tokio/tokio-stream/src/stream_map.rs",
        "target_symbols": [
            "poll_next_many"
        ],
        "expected_snippet_keywords": [
            "poll_next_many",
            "fn"
        ]
    },
    {
        "id": "tokio_11",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the FastRand struct in Tokio and how it handles <http://simul.iro.umontreal.ca/testu01/tu01.html>.",
        "target_file": "tokio/tokio-stream/src/stream_map.rs",
        "target_symbols": [
            "FastRand"
        ],
        "expected_snippet_keywords": [
            "FastRand",
            "struct"
        ]
    },
    {
        "id": "tokio_12",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the ReceiverStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/mpsc_bounded.rs",
        "target_symbols": [
            "ReceiverStream"
        ],
        "expected_snippet_keywords": [
            "ReceiverStream",
            "struct"
        ]
    },
    {
        "id": "tokio_13",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the close fn in Tokio and how it handles /// [`permit`]: struct@tokio::sync::mpsc::permit.",
        "target_file": "tokio/tokio-stream/src/wrappers/mpsc_bounded.rs",
        "target_symbols": [
            "close"
        ],
        "expected_snippet_keywords": [
            "close",
            "fn"
        ]
    },
    {
        "id": "tokio_14",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the size_hint fn in Tokio and how it handles /// [`permit`]: struct@tokio::sync::mpsc::permit.",
        "target_file": "tokio/tokio-stream/src/wrappers/mpsc_bounded.rs",
        "target_symbols": [
            "size_hint"
        ],
        "expected_snippet_keywords": [
            "size_hint",
            "fn"
        ]
    },
    {
        "id": "tokio_15",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the CtrlBreakStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/signal_windows.rs",
        "target_symbols": [
            "CtrlBreakStream"
        ],
        "expected_snippet_keywords": [
            "CtrlBreakStream",
            "struct"
        ]
    },
    {
        "id": "tokio_16",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the UnboundedReceiverStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/mpsc_unbounded.rs",
        "target_symbols": [
            "UnboundedReceiverStream"
        ],
        "expected_snippet_keywords": [
            "UnboundedReceiverStream",
            "struct"
        ]
    },
    {
        "id": "tokio_17",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the IntervalStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/interval.rs",
        "target_symbols": [
            "IntervalStream"
        ],
        "expected_snippet_keywords": [
            "IntervalStream",
            "struct"
        ]
    },
    {
        "id": "tokio_18",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the TcpListenerStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/tcp_listener.rs",
        "target_symbols": [
            "TcpListenerStream"
        ],
        "expected_snippet_keywords": [
            "TcpListenerStream",
            "struct"
        ]
    },
    {
        "id": "tokio_19",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the JoinSetStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/task.rs",
        "target_symbols": [
            "JoinSetStream"
        ],
        "expected_snippet_keywords": [
            "JoinSetStream",
            "struct"
        ]
    },
    {
        "id": "tokio_20",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the UnixListenerStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/unix_listener.rs",
        "target_symbols": [
            "UnixListenerStream"
        ],
        "expected_snippet_keywords": [
            "UnixListenerStream",
            "struct"
        ]
    },
    {
        "id": "tokio_21",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the WatchStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/watch.rs",
        "target_symbols": [
            "WatchStream"
        ],
        "expected_snippet_keywords": [
            "WatchStream",
            "struct"
        ]
    },
    {
        "id": "tokio_22",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the from_changes fn in Tokio and how it handles create a new `watchstream` that waits for the value to be changed..",
        "target_file": "tokio/tokio-stream/src/wrappers/watch.rs",
        "target_symbols": [
            "from_changes"
        ],
        "expected_snippet_keywords": [
            "from_changes",
            "fn"
        ]
    },
    {
        "id": "tokio_23",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the SignalStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/signal_unix.rs",
        "target_symbols": [
            "SignalStream"
        ],
        "expected_snippet_keywords": [
            "SignalStream",
            "struct"
        ]
    },
    {
        "id": "tokio_24",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the BroadcastStream struct in Tokio and how it handles [`stream`]: trait@futures_core::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/broadcast.rs",
        "target_symbols": [
            "BroadcastStream"
        ],
        "expected_snippet_keywords": [
            "BroadcastStream",
            "struct"
        ]
    },
    {
        "id": "tokio_25",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the BroadcastStreamRecvError enum in Tokio and how it handles an error returned from the inner stream of a [`broadcaststream`]..",
        "target_file": "tokio/tokio-stream/src/wrappers/broadcast.rs",
        "target_symbols": [
            "BroadcastStreamRecvError"
        ],
        "expected_snippet_keywords": [
            "BroadcastStreamRecvError",
            "enum"
        ]
    },
    {
        "id": "tokio_26",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the ReadDirStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/read_dir.rs",
        "target_symbols": [
            "ReadDirStream"
        ],
        "expected_snippet_keywords": [
            "ReadDirStream",
            "struct"
        ]
    },
    {
        "id": "tokio_27",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the LinesStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/lines.rs",
        "target_symbols": [
            "LinesStream"
        ],
        "expected_snippet_keywords": [
            "LinesStream",
            "struct"
        ]
    },
    {
        "id": "tokio_28",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the as_pin_mut fn in Tokio and how it handles obtain a pinned reference to the inner `lines<r>`..",
        "target_file": "tokio/tokio-stream/src/wrappers/lines.rs",
        "target_symbols": [
            "as_pin_mut"
        ],
        "expected_snippet_keywords": [
            "as_pin_mut",
            "fn"
        ]
    },
    {
        "id": "tokio_29",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the SplitStream struct in Tokio and how it handles [`stream`]: trait@crate::stream.",
        "target_file": "tokio/tokio-stream/src/wrappers/split.rs",
        "target_symbols": [
            "SplitStream"
        ],
        "expected_snippet_keywords": [
            "SplitStream",
            "struct"
        ]
    },
    {
        "id": "tokio_30",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Timeout struct in Tokio and how it handles stream returned by the [`timeout`](super::streamext::timeout) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/timeout.rs",
        "target_symbols": [
            "Timeout"
        ],
        "expected_snippet_keywords": [
            "Timeout",
            "struct"
        ]
    },
    {
        "id": "tokio_31",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Elapsed struct in Tokio and how it handles error returned by `timeout` and `timeoutrepeating`..",
        "target_file": "tokio/tokio-stream/src/stream_ext/timeout.rs",
        "target_symbols": [
            "Elapsed"
        ],
        "expected_snippet_keywords": [
            "Elapsed",
            "struct"
        ]
    },
    {
        "id": "tokio_32",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the TakeWhile struct in Tokio and how it handles stream for the [`take_while`](super::streamext::take_while) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/take_while.rs",
        "target_symbols": [
            "TakeWhile"
        ],
        "expected_snippet_keywords": [
            "TakeWhile",
            "struct"
        ]
    },
    {
        "id": "tokio_33",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the get_ref fn in Tokio and how it handles returns a reference to the inner stream..",
        "target_file": "tokio/tokio-stream/src/stream_ext/take_while.rs",
        "target_symbols": [
            "get_ref"
        ],
        "expected_snippet_keywords": [
            "get_ref",
            "fn"
        ]
    },
    {
        "id": "tokio_34",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the get_mut fn in Tokio and how it handles /// mutating the inner stream may confuse this combinator..",
        "target_file": "tokio/tokio-stream/src/stream_ext/take_while.rs",
        "target_symbols": [
            "get_mut"
        ],
        "expected_snippet_keywords": [
            "get_mut",
            "fn"
        ]
    },
    {
        "id": "tokio_35",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the get_pin_mut fn in Tokio and how it handles /// mutating the inner stream may confuse this combinator..",
        "target_file": "tokio/tokio-stream/src/stream_ext/take_while.rs",
        "target_symbols": [
            "get_pin_mut"
        ],
        "expected_snippet_keywords": [
            "get_pin_mut",
            "fn"
        ]
    },
    {
        "id": "tokio_36",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the AllFuture struct in Tokio and how it handles future for the [`all`](super::streamext::all) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/all.rs",
        "target_symbols": [
            "AllFuture"
        ],
        "expected_snippet_keywords": [
            "AllFuture",
            "struct"
        ]
    },
    {
        "id": "tokio_37",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Chain struct in Tokio and how it handles stream returned by the [`chain`](super::streamext::chain) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/chain.rs",
        "target_symbols": [
            "Chain"
        ],
        "expected_snippet_keywords": [
            "Chain",
            "struct"
        ]
    },
    {
        "id": "tokio_38",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the SkipWhile struct in Tokio and how it handles stream for the [`skip_while`](super::streamext::skip_while) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/skip_while.rs",
        "target_symbols": [
            "SkipWhile"
        ],
        "expected_snippet_keywords": [
            "SkipWhile",
            "struct"
        ]
    },
    {
        "id": "tokio_39",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the TimeoutRepeating struct in Tokio and how it handles stream returned by the [`timeout_repeating`](super::streamext::timeout_repeating) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/timeout_repeating.rs",
        "target_symbols": [
            "TimeoutRepeating"
        ],
        "expected_snippet_keywords": [
            "TimeoutRepeating",
            "struct"
        ]
    },
    {
        "id": "tokio_40",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Peekable struct in Tokio and how it handles stream returned by the [`peekable`](super::streamext::peekable) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/peekable.rs",
        "target_symbols": [
            "Peekable"
        ],
        "expected_snippet_keywords": [
            "Peekable",
            "struct"
        ]
    },
    {
        "id": "tokio_41",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the poll_peek fn in Tokio and how it handles poll to peek at the next item in the stream as a mutable reference..",
        "target_file": "tokio/tokio-stream/src/stream_ext/peekable.rs",
        "target_symbols": [
            "poll_peek"
        ],
        "expected_snippet_keywords": [
            "poll_peek",
            "fn"
        ]
    },
    {
        "id": "tokio_42",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Skip struct in Tokio and how it handles stream for the [`skip`](super::streamext::skip) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/skip.rs",
        "target_symbols": [
            "Skip"
        ],
        "expected_snippet_keywords": [
            "Skip",
            "struct"
        ]
    },
    {
        "id": "tokio_43",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Merge struct in Tokio and how it handles stream returned by the [`merge`](super::streamext::merge) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/merge.rs",
        "target_symbols": [
            "Merge"
        ],
        "expected_snippet_keywords": [
            "Merge",
            "struct"
        ]
    },
    {
        "id": "tokio_44",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Collect struct in Tokio and how it handles future returned by the [`collect`](super::streamext::collect) method..",
        "target_file": "tokio/tokio-stream/src/stream_ext/collect.rs",
        "target_symbols": [
            "Collect"
        ],
        "expected_snippet_keywords": [
            "Collect",
            "struct"
        ]
    },
    {
        "id": "tokio_45",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the FromStream trait in Tokio and how it handles enhancements to the rust language..",
        "target_file": "tokio/tokio-stream/src/stream_ext/collect.rs",
        "target_symbols": [
            "FromStream"
        ],
        "expected_snippet_keywords": [
            "FromStream",
            "trait"
        ]
    },
    {
        "id": "tokio_46",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the initialize fn in Tokio and how it handles initialize the collection.",
        "target_file": "tokio/tokio-stream/src/stream_ext/collect.rs",
        "target_symbols": [
            "initialize"
        ],
        "expected_snippet_keywords": [
            "initialize",
            "fn"
        ]
    },
    {
        "id": "tokio_47",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the extend fn in Tokio and how it handles /// return `true` to continue streaming, `false` complete collection..",
        "target_file": "tokio/tokio-stream/src/stream_ext/collect.rs",
        "target_symbols": [
            "extend"
        ],
        "expected_snippet_keywords": [
            "extend",
            "fn"
        ]
    },
    {
        "id": "tokio_48",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the finalize fn in Tokio and how it handles finalize collection into target type..",
        "target_file": "tokio/tokio-stream/src/stream_ext/collect.rs",
        "target_symbols": [
            "finalize"
        ],
        "expected_snippet_keywords": [
            "finalize",
            "fn"
        ]
    },
    {
        "id": "tokio_49",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the TryNext struct in Tokio and how it handles so dropping it will never lose a value..",
        "target_file": "tokio/tokio-stream/src/stream_ext/try_next.rs",
        "target_symbols": [
            "TryNext"
        ],
        "expected_snippet_keywords": [
            "TryNext",
            "struct"
        ]
    },
    {
        "id": "tokio_50",
        "repo": "tokio",
        "language": "Rust",
        "category": "Systems Runtime Navigation",
        "prompt": "Locate the Throttle struct in Tokio and how it handles implement `unpin` you can pin your throttle like this: `box::pin(your_throttle)`..",
        "target_file": "tokio/tokio-stream/src/stream_ext/throttle.rs",
        "target_symbols": [
            "Throttle"
        ],
        "expected_snippet_keywords": [
            "Throttle",
            "struct"
        ]
    },
    {
        "id": "fastapi_01",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_upload_file_invalid_pydantic_v2 defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_datastructures.py",
        "target_symbols": [
            "test_upload_file_invalid_pydantic_v2"
        ],
        "expected_snippet_keywords": [
            "test_upload_file_invalid_pydantic_v2"
        ]
    },
    {
        "id": "fastapi_02",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_default_placeholder_equals defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_datastructures.py",
        "target_symbols": [
            "test_default_placeholder_equals"
        ],
        "expected_snippet_keywords": [
            "test_default_placeholder_equals"
        ]
    },
    {
        "id": "fastapi_03",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_default_placeholder_bool defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_datastructures.py",
        "target_symbols": [
            "test_default_placeholder_bool"
        ],
        "expected_snippet_keywords": [
            "test_default_placeholder_bool"
        ]
    },
    {
        "id": "fastapi_04",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_upload_file_is_closed defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_datastructures.py",
        "target_symbols": [
            "test_upload_file_is_closed"
        ],
        "expected_snippet_keywords": [
            "test_upload_file_is_closed"
        ]
    },
    {
        "id": "fastapi_05",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is create_upload_file defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_datastructures.py",
        "target_symbols": [
            "create_upload_file"
        ],
        "expected_snippet_keywords": [
            "create_upload_file"
        ]
    },
    {
        "id": "fastapi_06",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_upload_file defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_datastructures.py",
        "target_symbols": [
            "test_upload_file"
        ],
        "expected_snippet_keywords": [
            "test_upload_file"
        ]
    },
    {
        "id": "fastapi_07",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is read_item defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "read_item"
        ],
        "expected_snippet_keywords": [
            "read_item"
        ]
    },
    {
        "id": "fastapi_08",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is no_body_status_code_exception defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "no_body_status_code_exception"
        ],
        "expected_snippet_keywords": [
            "no_body_status_code_exception"
        ]
    },
    {
        "id": "fastapi_09",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is no_body_status_code_with_detail_exception defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "no_body_status_code_with_detail_exception"
        ],
        "expected_snippet_keywords": [
            "no_body_status_code_with_detail_exception"
        ]
    },
    {
        "id": "fastapi_10",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is read_starlette_item defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "read_starlette_item"
        ],
        "expected_snippet_keywords": [
            "read_starlette_item"
        ]
    },
    {
        "id": "fastapi_11",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_get_item defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "test_get_item"
        ],
        "expected_snippet_keywords": [
            "test_get_item"
        ]
    },
    {
        "id": "fastapi_12",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_get_item_not_found defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "test_get_item_not_found"
        ],
        "expected_snippet_keywords": [
            "test_get_item_not_found"
        ]
    },
    {
        "id": "fastapi_13",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_get_starlette_item defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "test_get_starlette_item"
        ],
        "expected_snippet_keywords": [
            "test_get_starlette_item"
        ]
    },
    {
        "id": "fastapi_14",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_get_starlette_item_not_found defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "test_get_starlette_item_not_found"
        ],
        "expected_snippet_keywords": [
            "test_get_starlette_item_not_found"
        ]
    },
    {
        "id": "fastapi_15",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_no_body_status_code_exception_handlers defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "test_no_body_status_code_exception_handlers"
        ],
        "expected_snippet_keywords": [
            "test_no_body_status_code_exception_handlers"
        ]
    },
    {
        "id": "fastapi_16",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_no_body_status_code_with_detail_exception_handlers defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "test_no_body_status_code_with_detail_exception_handlers"
        ],
        "expected_snippet_keywords": [
            "test_no_body_status_code_with_detail_exception_handlers"
        ]
    },
    {
        "id": "fastapi_17",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_openapi_schema defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_starlette_exception.py",
        "target_symbols": [
            "test_openapi_schema"
        ],
        "expected_snippet_keywords": [
            "test_openapi_schema"
        ]
    },
    {
        "id": "fastapi_18",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is Item defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_multi_body_errors.py",
        "target_symbols": [
            "Item"
        ],
        "expected_snippet_keywords": [
            "Item"
        ]
    },
    {
        "id": "fastapi_19",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is save_item_no_body defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_multi_body_errors.py",
        "target_symbols": [
            "save_item_no_body"
        ],
        "expected_snippet_keywords": [
            "save_item_no_body"
        ]
    },
    {
        "id": "fastapi_20",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_put_correct_body defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_multi_body_errors.py",
        "target_symbols": [
            "test_put_correct_body"
        ],
        "expected_snippet_keywords": [
            "test_put_correct_body"
        ]
    },
    {
        "id": "fastapi_21",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_jsonable_encoder_requiring_error defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_multi_body_errors.py",
        "target_symbols": [
            "test_jsonable_encoder_requiring_error"
        ],
        "expected_snippet_keywords": [
            "test_jsonable_encoder_requiring_error"
        ]
    },
    {
        "id": "fastapi_22",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_put_incorrect_body_multiple defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_multi_body_errors.py",
        "target_symbols": [
            "test_put_incorrect_body_multiple"
        ],
        "expected_snippet_keywords": [
            "test_put_incorrect_body_multiple"
        ]
    },
    {
        "id": "fastapi_23",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is read_current_user defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_security_http_digest_description.py",
        "target_symbols": [
            "read_current_user"
        ],
        "expected_snippet_keywords": [
            "read_current_user"
        ]
    },
    {
        "id": "fastapi_24",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_security_http_digest defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_security_http_digest_description.py",
        "target_symbols": [
            "test_security_http_digest"
        ],
        "expected_snippet_keywords": [
            "test_security_http_digest"
        ]
    },
    {
        "id": "fastapi_25",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_security_http_digest_no_credentials defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_security_http_digest_description.py",
        "target_symbols": [
            "test_security_http_digest_no_credentials"
        ],
        "expected_snippet_keywords": [
            "test_security_http_digest_no_credentials"
        ]
    },
    {
        "id": "fastapi_26",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_security_http_digest_incorrect_scheme_credentials defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_security_http_digest_description.py",
        "target_symbols": [
            "test_security_http_digest_incorrect_scheme_credentials"
        ],
        "expected_snippet_keywords": [
            "test_security_http_digest_incorrect_scheme_credentials"
        ]
    },
    {
        "id": "fastapi_27",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is form_json_list defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_json_type.py",
        "target_symbols": [
            "form_json_list"
        ],
        "expected_snippet_keywords": [
            "form_json_list"
        ]
    },
    {
        "id": "fastapi_28",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is query_json_list defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_json_type.py",
        "target_symbols": [
            "query_json_list"
        ],
        "expected_snippet_keywords": [
            "query_json_list"
        ]
    },
    {
        "id": "fastapi_29",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is header_json_list defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_json_type.py",
        "target_symbols": [
            "header_json_list"
        ],
        "expected_snippet_keywords": [
            "header_json_list"
        ]
    },
    {
        "id": "fastapi_30",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is cookie_json_list defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_json_type.py",
        "target_symbols": [
            "cookie_json_list"
        ],
        "expected_snippet_keywords": [
            "cookie_json_list"
        ]
    },
    {
        "id": "fastapi_31",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_form_json_list defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_json_type.py",
        "target_symbols": [
            "test_form_json_list"
        ],
        "expected_snippet_keywords": [
            "test_form_json_list"
        ]
    },
    {
        "id": "fastapi_32",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_query_json_list defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_json_type.py",
        "target_symbols": [
            "test_query_json_list"
        ],
        "expected_snippet_keywords": [
            "test_query_json_list"
        ]
    },
    {
        "id": "fastapi_33",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_header_json_list defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_json_type.py",
        "target_symbols": [
            "test_header_json_list"
        ],
        "expected_snippet_keywords": [
            "test_header_json_list"
        ]
    },
    {
        "id": "fastapi_34",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_cookie_json_list defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_json_type.py",
        "target_symbols": [
            "test_cookie_json_list"
        ],
        "expected_snippet_keywords": [
            "test_cookie_json_list"
        ]
    },
    {
        "id": "fastapi_35",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is client_fixture defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_union_body_discriminator_annotated.py",
        "target_symbols": [
            "client_fixture"
        ],
        "expected_snippet_keywords": [
            "client_fixture"
        ]
    },
    {
        "id": "fastapi_36",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is get_pet_type defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_union_body_discriminator_annotated.py",
        "target_symbols": [
            "get_pet_type"
        ],
        "expected_snippet_keywords": [
            "get_pet_type"
        ]
    },
    {
        "id": "fastapi_37",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is create_pet_assignment defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_union_body_discriminator_annotated.py",
        "target_symbols": [
            "create_pet_assignment"
        ],
        "expected_snippet_keywords": [
            "create_pet_assignment"
        ]
    },
    {
        "id": "fastapi_38",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is create_pet_annotated defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_union_body_discriminator_annotated.py",
        "target_symbols": [
            "create_pet_annotated"
        ],
        "expected_snippet_keywords": [
            "create_pet_annotated"
        ]
    },
    {
        "id": "fastapi_39",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_union_body_discriminator_assignment defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_union_body_discriminator_annotated.py",
        "target_symbols": [
            "test_union_body_discriminator_assignment"
        ],
        "expected_snippet_keywords": [
            "test_union_body_discriminator_assignment"
        ]
    },
    {
        "id": "fastapi_40",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_union_body_discriminator_annotated defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_union_body_discriminator_annotated.py",
        "target_symbols": [
            "test_union_body_discriminator_annotated"
        ],
        "expected_snippet_keywords": [
            "test_union_body_discriminator_annotated"
        ]
    },
    {
        "id": "fastapi_41",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is get_client defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_computed_fields.py",
        "target_symbols": [
            "get_client"
        ],
        "expected_snippet_keywords": [
            "get_client"
        ]
    },
    {
        "id": "fastapi_42",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is Rectangle defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_computed_fields.py",
        "target_symbols": [
            "Rectangle"
        ],
        "expected_snippet_keywords": [
            "Rectangle"
        ]
    },
    {
        "id": "fastapi_43",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is area defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_computed_fields.py",
        "target_symbols": [
            "area"
        ],
        "expected_snippet_keywords": [
            "area"
        ]
    },
    {
        "id": "fastapi_44",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is read_root defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_computed_fields.py",
        "target_symbols": [
            "read_root"
        ],
        "expected_snippet_keywords": [
            "read_root"
        ]
    },
    {
        "id": "fastapi_45",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is read_responses defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_computed_fields.py",
        "target_symbols": [
            "read_responses"
        ],
        "expected_snippet_keywords": [
            "read_responses"
        ]
    },
    {
        "id": "fastapi_46",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_get defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_computed_fields.py",
        "target_symbols": [
            "test_get"
        ],
        "expected_snippet_keywords": [
            "test_get"
        ]
    },
    {
        "id": "fastapi_47",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is User defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_security_oauth2.py",
        "target_symbols": [
            "User"
        ],
        "expected_snippet_keywords": [
            "User"
        ]
    },
    {
        "id": "fastapi_48",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is get_current_user defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_security_oauth2.py",
        "target_symbols": [
            "get_current_user"
        ],
        "expected_snippet_keywords": [
            "get_current_user"
        ]
    },
    {
        "id": "fastapi_49",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is login defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_security_oauth2.py",
        "target_symbols": [
            "login"
        ],
        "expected_snippet_keywords": [
            "login"
        ]
    },
    {
        "id": "fastapi_50",
        "repo": "fastapi",
        "language": "Python",
        "category": "Web Framework Architecture",
        "prompt": "Where is test_security_oauth2 defined in FastAPI and what are its signature and dependencies?",
        "target_file": "fastapi/tests/test_security_oauth2.py",
        "target_symbols": [
            "test_security_oauth2"
        ],
        "expected_snippet_keywords": [
            "test_security_oauth2"
        ]
    }
]

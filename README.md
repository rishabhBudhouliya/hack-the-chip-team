# Hack the Chip team

Code and notes for the NYU × Annapurna Labs Hack the Chip event: agent loops on AWS Trainium.

| Folder | What it is |
|---|---|
| [`roofline/`](roofline/) | Works out the fastest an operation can run on Trainium2 (softmax, the AlexNet conv block, matmul) and compares a kernel's predicted latency against it |

Everything runs inside a seat pod. Copy a folder in with
`kubectl cp <folder> seat-<N>:/root/<folder> -c app`. Don't commit AWS credentials.

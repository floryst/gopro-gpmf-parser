# Building

```
podman build -t flatbuffers:24.3.25 .
```

# Running

```
podman run --rm -it -v "$PWD":/src flatbuffers:24.3.25 /usr/bin/flatc --python ./schema.fbs
```
```
usage: client.py [-h] [-m MODEL] [-u URL] [-i] [-v] [-t CLIENT_TIMEOUT] [-s]
                 [-r ROOT_CERTIFICATES] [-p PRIVATE_KEY]
                 [-x CERTIFICATE_CHAIN]
                 {dummy,image,video}

positional arguments:
  {dummy,image,video}   Run mode. 'dummy' will send an emtpy buffer to the
                        server to test if inference works. 'image' will
                        process an image. 'video' will process a video.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Inference model name. Default is yolov4.
  -u URL, --url URL     Inference server URL. Default is localhost:8001.
  -i, --model-info      Print model status, configuration and statistics
  -v, --verbose         Enable verbose client output
  -t CLIENT_TIMEOUT, --client-timeout CLIENT_TIMEOUT
                        Client timeout in seconds. Default is None.
  -s, --ssl             Enable SSL encrypted channel to the server
  -r ROOT_CERTIFICATES, --root-certificates ROOT_CERTIFICATES
                        File holding PEM-encoded root certificates. Default is
                        None.
  -p PRIVATE_KEY, --private-key PRIVATE_KEY
                        File holding PEM-encoded private key. Default is None.
  -x CERTIFICATE_CHAIN, --certificate-chain CERTIFICATE_CHAIN
                        File holding PEM-encoded certicate chain. Default is
                        None.
```
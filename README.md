# Object-Detector-App

A real-time object recognition application using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).

## Getting Started
1. `conda env create -f environment.yml`
2. `python object_detection_app.py`
    Optional arguments (default value):
    * Show all commands `--help`
    * Device index of the camera `--source=0`
    * Width of the frames in the video stream `--width=480`
    * Height of the frames in the video stream `--height=360`
    * Number of workers `--num-workers=2`
    * Size of the queue `--queue-size=5`
    * URL for video stream `--url=<rstp://...>`
    * Turn on GUI (defaulted to run headless) `--gui`
    * Turn on vocal commands on MacOS (defaulted to silent) `--say`
    * State Buffer Size, how many "states" to capture `--state-queue-size=5`

## Development
### Updating the environment
`conda env update -f environment.yml`

### Tests
```
pytest -vs utils/
python -m pytest
python -m unittest discover -s object_detection -p "*_test.py"
```

### Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)

### API 
Our API is accessible via the MQTT protocol.

#### `dev/chloe/explorer/statement`
We subscribe to a topic coming from an Android client. Incoming messages should be encoded as JSON objects that match
the following format: 

```json
{
  "messageId": 123,
  "serviceId": 53453,
  "userId": 4823942,
  "timestamp": 1234123412341234,
  "statement": "describe what is going on around me" 
  //...
}
```

#### `dev/chloe/response/<userid>/<action>`
We publish to the root topic `dev/chloe/response` via subtopics scoped by the end user's id and the desired action. For instance, if we expect the client with id `1324234` to read the text response aloud (i.e. the `say` action), we will publish to the following topic path: `dev/chloe/response/1324234/say`. 

Messages should be encoded as JSON objects in the following format: 

```json
{
  "messageId": 124, // ID for the current payload
  "statementId": 123, // ID of statement payload (payload this is in response to, see above)
  "timestamp": 1234123412341234,

  // Was the service successful?
  "status": {
    "code": "ch-vis-000", // <project code>-<module code>-<error/status code>
    "message": "success" 
  },
  
  "action": "say",
  "args": ["arg1", "arg2", "arg3"], // **Prefer kwargs to args**
  "kwargs": {
    "confidence": 0.87,  // argument that should always be present
    "key1": 1,
    "key2": "kwarg2"
  },
  
  //...
}
```

TODO(Alex) Revise
Here is an example of a response for "say":
Topic: `nsf/ai/say`
Payload: 
```json
{
  "messageId": 124, 
  "statementId": 123,
  "timestamp": 1234123412341234,
  "status": {
    "code": "ch-vis-000",
    "message": "success" 
  },
  "action": "say",
  "args": [], 
  "kwargs": {
    "confidence": 0.87,
    "text": "there is 1 person and a chair around you",
    "wordsPerMin": 200,
    "voiceGender": "Female"
  }
}
```

### Agent-Chloe Experiment Configuration Discussion
- Should be configured on dashboard. 
- Response to explorer should have a delay, whether they come from Chloe or the AI. The explorer should not be able to distinguish between human and machine. 
- Want to design intentional fallback from the AI to the Human agent. Thus, we need two buttons: the random send (either AI or Human), and a **SEND!** that forcibly sends the human response over the AI. 

## Notes
- ~~OpenCV 3.1 might crash on OSX after a while, so that's why I had to switch to version 3.0. See open issue and solution [here](https://github.com/opencv/opencv/issues/5874).~~
- Moving the `.read()` part of the video stream in a multiple child processes did not work. However, it was possible to move it to a separate thread.

## Copyright
See [LICENSE](LICENSE) for details.

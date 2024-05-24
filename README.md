### Setup

1. Install ROS Noetic http://wiki.ros.org/noetic/Installation/
2. Install cv_bridge ROS package http://wiki.ros.org/cv_bridge
3. Setup a virtual environment for Python containing PyTorch with CUDA installed https://www.cherryservers.com/blog/how-to-install-pytorch-ubuntu
4. Install Ultralytics package inside the virtual environment https://docs.ultralytics.com/quickstart/#install-ultralytics
5. Install other needed packages such as SciPy if it isn't installed by Ultralytics

### Understanding the codebase
The codebase is simple. The only interesting executable is the tracker node `tracker_node.py`. It handles the object tracking and object localization.

Example output 
```
[
  {
    'class': 1,
    'label': 'box',
    'confidence': 0.9919964671134949,
    'tracking_id': 1,
    'boundingBox': {
      'x': 155.87393188476562,
      'y': 324.0987548828125,
      'w': 170.0921630859375,
      'h': 163.91943359375
    },
    'worldCoordinates': {
      'x': -0.12249637378623776,
      'y': -0.9434229810348629,
      'z': 0.5227058974543436
    }
  },
  {
    'class': 1,
    'label': 'box',
    'confidence': 0.9759730100631714,
    'tracking_id': 2,
    'boundingBox': {
      'x': 428.02325439453125,
      'y': 324.4464416503906,
      'w': 162.5687255859375,
      'h': 173.02020263671875
    },
    'worldCoordinates': {
      'x': 0.16471446981522775,
      'y': -0.943118324301396,
      'z': 0.7662337043024138
    }
  },
  {
    'class': 0,
    'label': 'tennis ball',
    'confidence': 0.9518796801567078,
    'tracking_id': 3,
    'boundingBox': {
      'x': 356.22052001953125,
      'y': 418.6640625,
      'w': 49.88037109375,
      'h': 50.977294921875
    },
    'worldCoordinates': {
      'x': 0.17569001047898541,
      'y': -0.9607874463339089,
      'z': 0.6014914712242809
    }
  },
  {
    'class': 0,
    'label': 'tennis ball',
    'confidence': 0.9383163452148438,
    'tracking_id': 4,
    'boundingBox': {
      'x': 288.78265380859375,
      'y': 418.957275390625,
      'w': 47.27423095703125,
      'h': 50.3983154296875
    },
    'worldCoordinates': {
      'x': 0.10251490663872125,
      'y': -0.9610142073142107,
      'z': 0.5394111907845041
    }
  },
  {
    'class': 0,
    'label': 'tennis ball',
    'confidence': 0.9289605021476746,
    'tracking_id': 5,
    'boundingBox': {
      'x': 220.53787231445312,
      'y': 418.36798095703125,
      'w': 49.09814453125,
      'h': 50.59423828125
    },
    'worldCoordinates': {
      'x': 0.029302155009578404,
      'y': -0.9616382600418262,
      'z': 0.4772073988284648
    }
  }
]
```

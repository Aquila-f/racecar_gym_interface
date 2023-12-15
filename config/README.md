## YAML Configuration

This README provides an overview of the YAML configuration file used in the project. The YAML file is used to define various settings and parameters for the application.

### Structure

The YAML configuration file follows a key-value pair structure. Each key represents a specific setting or parameter, and its corresponding value defines the value for that setting.

### Example

Here's an example of a YAML configuration file:

#### 3 direction yaml file
#### motor: [-1, 1]
#### steering: [-1, 1]
0:
  key: ArrowLeft
  motor: 0.01
  steering: -1
1:
  key: ArrowUp
  motor: 0.01
  steering: 0
2:
  key: ArrowRight
  motor: 0.01
  steering: 1
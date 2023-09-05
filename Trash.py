# @app.route('/process_frame_butterfly', methods=['POST'])
# def butterfly():
#     # Extract the image from the request
#     image_data = request.form.get('image').split(',', 1)[1]
#     image_decoded = base64.b64decode(image_data)
#     image_np = np.frombuffer(image_decoded, dtype=np.uint8)
#     image = cv2.imdecode(image_np, flags=1)
#
#     imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(imgRGB)
#
#     # Initialize necessary variables and classes
#     graph = GraphPlotter(labels=["Right Elbow Angle", "Left Elbow Angle", "Right Arm Angle", "Left Arm Angle"])
#     curr_time = time.time()
#     count = 0
#     is_count = False
#     data1 = []
#
#     if results.pose_landmarks:
#         # Calculate the elbow and arm angles
#         right_elbow_angle, left_elbow_angle, right_arm_angle, left_arm_angle = angle.elbow_shoulder(results, mpPose)
#
#         graph_img = graph.render(image.shape)
#         combined_image = np.hstack((image, graph_img))
#
#         ret, buffer = cv2.imencode('.jpg', combined_image)
#         image_encoded = base64.b64encode(buffer).decode('utf-8')
#
#         graph.update_data(
#             **{
#                 "Right Elbow Angle": right_elbow_angle,
#                 "Left Elbow Angle": left_elbow_angle,
#                 "Right Arm Angle": right_arm_angle,
#                 "Left Arm Angle": left_arm_angle
#             }
#         )
#
#         # Check and update the count
#         if right_arm_angle > 25 and left_arm_angle > 25 and right_elbow_angle > 110 and left_elbow_angle > 110:
#             if not is_count:
#                 count += 1
#                 is_count = True
#         else:
#             is_count = False
#
#         # Update the JSON object with the angle values
#         angle_data = {
#             "time": datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S'),
#             "right_elbow_angle": right_elbow_angle,
#             "left_elbow_angle": left_elbow_angle,
#             "right_arm_angle": right_arm_angle,
#             "left_arm_angle": left_arm_angle,
#             "exercise_count": count
#         }
#         data1.append(angle_data)
#
#         # Save the JSON data to a file
#         json_filename = "angle_data.json"
#         with open(json_filename, "w") as json_file:
#             json.dump(data1, json_file, indent=4)
#
#         # Return the computed values to the client
#         return jsonify({
#             "right_elbow_angle": right_elbow_angle,
#             "left_elbow_angle": left_elbow_angle,
#             "right_arm_angle": right_arm_angle,
#             "left_arm_angle": left_arm_angle,
#             "exercise_count": count,
#             "image": "data:image/jpeg;base64," + image_encoded
#         })
#     else:
#         return jsonify({"error": "No landmarks detected"})


class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def add(self, value):
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data = self.data[1:]

    def get_average(self):
        if self.data:
            return sum(self.data) / len(self.data)
        else:
            return 0

class GraphPlotter:
    def __init__(self, labels, max_length=60):
        self.max_length = max_length
        self.fig, self.ax = plt.subplots(2, 2)
        self.data_queues = {label: self.init_queue() for label in labels}
        self.lines = {label: self.init_line(idx, label) for idx, label in enumerate(labels)}
        plt.tight_layout()

    def init_queue(self):
        return [0 for _ in range(self.max_length)]

    def init_line(self, idx, label):
        row, col = divmod(idx, 2)
        line, = self.ax[row, col].plot(np.linspace(0, self.max_length, self.max_length), self.data_queues[label])
        self.ax[row, col].set_title(label)
        return line

    def update_data(self, **kwargs):
        for label, value in kwargs.items():
            self.data_queues[label].append(value)
            if len(self.data_queues[label]) > self.max_length:
                self.data_queues[label].pop(0)
            self.lines[label].set_ydata(self.data_queues[label])
            self.ax[divmod(list(self.lines.keys()).index(label), 2)].set_ylim(min(self.data_queues[label]) - 10, max(self.data_queues[label]) + 10)

    def render(self, img_shape):
        self.fig.canvas.draw()
        img_data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_shape = self.fig.canvas.get_width_height()[::-1] + (3,)
        img_data = img_data.reshape(img_shape)
        desired_height = img_shape[0]
        desired_width = int(desired_height * img_shape[1] / img_shape[0])
        graph_img = cv2.resize(img_data, (desired_width, desired_height))
        return graph_img

    def close(self):
        plt.close(self.fig)

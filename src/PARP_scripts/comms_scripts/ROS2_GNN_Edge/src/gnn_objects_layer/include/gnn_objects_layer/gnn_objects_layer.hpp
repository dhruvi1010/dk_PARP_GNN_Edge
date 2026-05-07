#pragma once

#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav2_costmap_2d/costmap_layer.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "geometry_msgs/msg/polygon.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "gnn_interfaces/msg/tracked_polygon.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>



namespace gnn_objects_layer
{

class GNNObjectsLayer : public nav2_costmap_2d::Layer
{
public:
  GNNObjectsLayer();
  virtual ~GNNObjectsLayer() = default;
    virtual void reset() override {}

    virtual bool isClearable() override {
    return true;
    }

  virtual void onInitialize() override;

  virtual void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y,
    double * max_x, double * max_y) override;

  virtual void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

private:
  void trackedPolygonCallback(
    const gnn_interfaces::msg::TrackedPolygon::SharedPtr msg);

  std::vector<gnn_interfaces::msg::TrackedPolygon> tracked_polygons_;
  rclcpp::Subscription<gnn_interfaces::msg::TrackedPolygon>::SharedPtr sub_;
  std::mutex polygon_mutex_;
};

}  // namespace gnn_objects_layer

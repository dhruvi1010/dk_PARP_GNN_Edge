#include "gnn_objects_layer/gnn_objects_layer.hpp"

#include "pluginlib/class_list_macros.hpp"
#include "nav2_costmap_2d/costmap_math.hpp"
#include <algorithm>

PLUGINLIB_EXPORT_CLASS(gnn_objects_layer::GNNObjectsLayer, nav2_costmap_2d::Layer)

namespace gnn_objects_layer
{

GNNObjectsLayer::GNNObjectsLayer() {}

void GNNObjectsLayer::onInitialize()
{
    auto node = node_.lock();
    if (!node) {
    RCLCPP_ERROR(rclcpp::get_logger("GNNObjectsLayer"), "Node handle is null");
    return;
    }

  declareParameter("topic", rclcpp::ParameterValue("/tracked_polygons"));

  std::string topic;
  topic = node->declare_parameter<std::string>("topic", "/tracked_polygons");

  sub_ = node->create_subscription<gnn_interfaces::msg::TrackedPolygon>(
    topic, 10,
    std::bind(&GNNObjectsLayer::trackedPolygonCallback, this, std::placeholders::_1));

  enabled_ = true;
  tracked_polygons_.clear();
}

void GNNObjectsLayer::trackedPolygonCallback(
  const gnn_interfaces::msg::TrackedPolygon::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(polygon_mutex_);
  tracked_polygons_.push_back(*msg);
}

void GNNObjectsLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  std::lock_guard<std::mutex> lock(polygon_mutex_);

  for (const auto & poly : tracked_polygons_) {
    for (const auto & pt : poly.polygon.points) {
      *min_x = std::min(*min_x, static_cast<double>(pt.x));
      *min_y = std::min(*min_y, static_cast<double>(pt.y));
      *max_x = std::max(*max_x, static_cast<double>(pt.x));
      *max_y = std::max(*max_y, static_cast<double>(pt.y));
    }
  }
}

void GNNObjectsLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int /*min_i*/, int /*min_j*/, int /*max_i*/, int /*max_j*/)
{
  std::lock_guard<std::mutex> lock(polygon_mutex_);

  for (const auto & poly : tracked_polygons_) {
    const auto & points = poly.polygon.points;
    std::vector<cv::Point> contour;

    for (const auto & pt : points) {
      unsigned int mx, my;
      if (master_grid.worldToMap(pt.x, pt.y, mx, my)) {
        contour.emplace_back(mx, my);
      }
    }

    if (contour.size() >= 3) {
      // Create a blank grid and fill the polygon
      cv::Mat mask(master_grid.getSizeInCellsY(), master_grid.getSizeInCellsX(), CV_8UC1, cv::Scalar(0));
      std::vector<std::vector<cv::Point>> contours = {contour};
      cv::fillPoly(mask, contours, cv::Scalar(255));

      for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
          if (mask.at<uchar>(y, x) == 255) {
            master_grid.setCost(x, y, nav2_costmap_2d::LETHAL_OBSTACLE);
          }
        }
      }
    }
  }

  tracked_polygons_.clear();
}

}  // namespace gnn_objects_layer

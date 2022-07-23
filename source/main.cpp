#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <defer.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <ft2build.h>
#include FT_FREETYPE_H

#include "glm.hpp"

#include "shader_catalog.hpp"

#include "font.cpp"

struct Transform {
	float fovy         = glm::radians(60.0f);
	float distance     = 0.42f;
	glm::mat3 rotation = glm::mat3(1.0f);
	glm::vec3 position = glm::vec3(0.0f);

	glm::mat4 getProjectionMatrix(float aspect) {
		return glm::perspective(/* fovy = */ glm::radians(60.0f), aspect, 0.002f, 12.000f);
	}

	glm::mat4 getViewMatrix() {
		auto translation = glm::translate(position);
		return glm::lookAt(glm::vec3(0, 0, distance), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)) * glm::mat4(rotation) * translation;
	}
};

struct DragController {
	Transform* transform = nullptr;
	int activeButton = -1;

	double dragX, dragY;
	double wrapX, wrapY;
	double virtualX, virtualY;
	glm::vec3 dragTarget;

	bool unprojectMousePositionToXYPlane(GLFWwindow* window, double x, double y, glm::vec3& result) {
		int iwidth = 0, iheight = 0;
		glfwGetWindowSize(window, &iwidth, &iheight);

		double width = iwidth;
		double height = iheight;

		glm::mat4 projection = transform->getProjectionMatrix(float(width / height));
		glm::mat4 view = transform->getViewMatrix();

		double relX = x/width*2.0 - 1.0;
		double relY = y/height*2.0 - 1.0;

		glm::vec4 clipPos = glm::vec4(float(relX), -float(relY), 0.5f, 1.0f);
		glm::vec4 worldPos = glm::inverse(projection * view) * clipPos;
		worldPos *= 1.0f / worldPos.w;

		glm::vec3 pos = glm::vec3(glm::column(glm::inverse(view), 3));
		glm::vec3 dir = glm::normalize(glm::vec3(worldPos) - pos);
		float t = -pos.z / dir.z;

		result = pos + t * dir;
		return t > 0.0f;
	}

	void onMouseButton(GLFWwindow* window, int button, int action, int mods) {
		if (action == GLFW_PRESS && activeButton == -1) {
			activeButton = button;
			glfwGetCursorPos(window, &dragX, &dragY);
			wrapX = std::numeric_limits<double>::quiet_NaN();
			wrapY = std::numeric_limits<double>::quiet_NaN();
			virtualX = dragX;
			virtualY = dragY;

			glm::vec3 target;
			bool ok = unprojectMousePositionToXYPlane(window, dragX, dragY, target);
			dragTarget = ok ? target : glm::vec3();
		} else if (action == GLFW_RELEASE && activeButton == button) {
			activeButton = -1;
			dragX = 0.0;
			dragY = 0.0;
			wrapX = std::numeric_limits<double>::quiet_NaN();
			wrapY = std::numeric_limits<double>::quiet_NaN();
			virtualX = 0.0;
			virtualY = 0.0;
			dragTarget = glm::vec3();
		}
	}

	void onCursorPos(GLFWwindow* window, double x, double y) {
		if (activeButton == -1) return;

		int iwidth = 0, iheight = 0;
		glfwGetWindowSize(window, &iwidth, &iheight);

		double width = iwidth;
		double height = iheight;

		double deltaX = x-dragX;
		double deltaY = y-dragY;

		if (!std::isnan(wrapX) && !std::isnan(wrapY)) {
			double wrapDeltaX = x-wrapX;
			double wrapDeltaY = y-wrapY;
			if (wrapDeltaX*wrapDeltaX+wrapDeltaY*wrapDeltaY < deltaX*deltaX+deltaY*deltaY) {
				deltaX = wrapDeltaX;
				deltaY = wrapDeltaY;
				wrapX = std::numeric_limits<double>::quiet_NaN();
				wrapY = std::numeric_limits<double>::quiet_NaN();
			}
		}

		dragX = x;
		dragY = y;

		double targetX = x;
		double targetY = y;
		bool changed = false;
		if (targetX < 0) {
			targetX += width - 1;
			changed = true;
		} else if (targetX >= width) {
			targetX -= width - 1;
			changed = true;
		}
		if (targetY < 0) {
			targetY += height - 1;
			changed = true;
		} else if (targetY >= height) {
			targetY -= height - 1;
			changed = true;
		}
		if (changed) {
			glfwSetCursorPos(window, targetX, targetY);
			wrapX = targetX;
			wrapY = targetY;
		}

		if (activeButton == GLFW_MOUSE_BUTTON_2) {
			virtualX += deltaX;
			virtualY += deltaY;

			glm::vec3 target;
			bool ok = unprojectMousePositionToXYPlane(window, virtualX, virtualY, target);
			if (ok) {
				float x = transform->position.x;
				float y = transform->position.y;
				glm::vec3 delta = target - dragTarget;
				transform->position.x = glm::clamp(x + delta.x, -2.0f, 2.0f);
				transform->position.y = glm::clamp(y + delta.y, -2.0f, 2.0f);
			}
		} else if (activeButton == GLFW_MOUSE_BUTTON_3) {
			// Turntable rotation.
			double size = glm::min(width, height);
			glm::mat3 rx = glm::rotate(float(deltaX / size * glm::pi<double>()), glm::vec3(0, 0, 1));
			glm::mat3 ry = glm::rotate(float(deltaY / size * glm::pi<double>()), glm::vec3(1, 0, 0));
			transform->rotation = ry * transform->rotation * rx;
		} else {
			// Trackball rotation.
			double size = glm::min(width, height);
			glm::mat3 rx = glm::rotate(float(deltaX / size * glm::pi<double>()), glm::vec3(0, 1, 0));
			glm::mat3 ry = glm::rotate(float(deltaY / size * glm::pi<double>()), glm::vec3(1, 0, 0));
			transform->rotation = ry * rx * transform->rotation;
		}
	}

	void onScroll(GLFWwindow* window, double xOffset, double yOffset) {
		float factor = glm::clamp(1.0-float(yOffset)/10.0, 0.1, 1.9);
		transform->distance = glm::clamp(transform->distance * factor, 0.010f, 10.000f);
	}
};

namespace {
	FT_Library library;

	Transform transform;
	DragController dragController;

	// Empty VAO used when the vertex shader has no input and only uses gl_VertexID,
	// because OpenGL still requires a non-zero VAO to be bound for the draw call.
	GLuint emptyVAO;

	std::unique_ptr<ShaderCatalog> shaderCatalog;
	std::shared_ptr<ShaderCatalog::Entry> backgroundShader;
	std::shared_ptr<ShaderCatalog::Entry> fontShader;

	std::unique_ptr<Font> font;
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	dragController.onMouseButton(window, button, action, mods);
}

static void cursorPosCallback(GLFWwindow* window, double x, double y) {
	dragController.onCursorPos(window, x, y);
}

static void scrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
	dragController.onScroll(window, xOffset, yOffset);
}

static void dropCallback(GLFWwindow* window, int pathCount, const char* paths[]) {
	if (pathCount == 0) return;

	std::string filename = paths[0];

	std::string error;
	FT_Face face = Font::loadFace(library, filename, error);
	if (error != "") {
		std::cerr << "[font] failed to load " << filename << ": " << error << std::endl;
	} else {
		font = std::make_unique<Font>(face);
	}
}

int main(int argc, char* argv[]) {
	if (!glfwInit()) {
		std::cerr << "ERROR: failed to initialize GLFW" << std::endl;
		return 1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

	GLFWwindow* window = glfwCreateWindow(1600, 900, "GPU Font Rendering Demo", nullptr, nullptr);
	if (!window) {
		std::cerr << "ERROR: failed to create GLFW window" << std::endl;
		glfwTerminate();
		return 1;
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
		std::cerr << "ERROR: failed to initialize OpenGL context" << std::endl;
		glfwTerminate();
		return 1;
	}

	{
		FT_Error error = FT_Init_FreeType(&library);
		if (error) {
			std::cerr << "ERROR: failed to initialize FreeType" << std::endl;
			glfwTerminate();
			return 1;
		}
	}

	dragController.transform = &transform;
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);
	glfwSetScrollCallback(window, scrollCallback);
	glfwSetDropCallback(window, dropCallback);

	glGenVertexArrays(1, &emptyVAO);

	shaderCatalog = std::make_unique<ShaderCatalog>("shaders");
	backgroundShader = shaderCatalog->get("background");
	fontShader = shaderCatalog->get("font");

	{
		std::string filename = "fonts/SourceSerifPro-Regular.otf";

		std::string error;
		FT_Face face = Font::loadFace(library, filename, error);
		if (error != "") {
			std::cerr << "[font] failed to load " << filename << ": " << error << std::endl;
		} else {
			font = std::make_unique<Font>(face);
		}
	}

	while(!glfwWindowShouldClose(window)) {
		shaderCatalog->update();

		glfwPollEvents();

		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		GLuint location;

		glm::mat4 projection = transform.getProjectionMatrix((float)width / height);
		glm::mat4 view = transform.getViewMatrix();
		glm::mat4 model = glm::mat4(1.0f);

		{ // Draw background.
			GLuint program = backgroundShader->program;
			glUseProgram(program);
			glBindVertexArray(emptyVAO);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
			glBindVertexArray(0);
			glUseProgram(0);
		}

		// Uses premultiplied-alpha.
		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

		if (font) {
			GLuint program = fontShader->program;
			glUseProgram(program);

			font->program = program;
			font->dilation = 0.1f;
			font->worldSize = 0.2f;
			font->drawSetup();

			location = glGetUniformLocation(program, "projection");
			glUniformMatrix4fv(location, 1, false, glm::value_ptr(projection));
			location = glGetUniformLocation(program, "view");
			glUniformMatrix4fv(location, 1, false, glm::value_ptr(view));
			location = glGetUniformLocation(program, "model");
			glUniformMatrix4fv(location, 1, false, glm::value_ptr(model));

			location = glGetUniformLocation(program, "color");
			glUniform4f(location, 1.0f, 1.0f, 1.0f, 1.0f);

			font->draw(0, 0, "Hello, world!");
			glUseProgram(0);
		}

		glDisable(GL_BLEND);

		glfwSwapBuffers(window);
	}

	glfwTerminate();
	return 0;
}

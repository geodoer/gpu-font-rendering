#version 330 core

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

layout (location = 0) in vec2 vertexPosition;
layout (location = 1) in vec2 vertexUV;
layout (location = 2) in int  vertexIndex;

out vec2 uv;
flat out int bufferIndex;
//flat声明一个标量或向量变量是“平面的”（即不会被插值）

void main() {
	gl_Position = projection * view * model * vec4(vertexPosition, 0, 1);
	uv = vertexUV;
	bufferIndex = vertexIndex;
}

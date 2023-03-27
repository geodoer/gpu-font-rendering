#pragma once

#include <string>
#include <memory>

// A shader catalog loads and compiles shaders from a directory. Vertex and
// fragment shaders are matched based on their filename (e.g. example.vert and
// example.frag are loaded and linked together to form the "example" program).
// Whenever a shader file changes on disk, the corresponding program is
// recompiled and relinked.
/*
	ShaderCatalog(着色器目录管理者)会从目录中加载和编译着色器
	例如：传入"example"，则加载"example.vert"作为顶点着色器、"examples.frag"作为片元着色器
	当着色器文件在磁盘上发生改变时，对应的程序将自动重新编译与链接（内部会监听文件变化）
 */
class ShaderCatalog {
public:
	struct Entry {
		unsigned int program;

		Entry() : program(0) {}
		Entry(unsigned int program) : program(program) {}
	};

	ShaderCatalog(const std::string& dir);
	~ShaderCatalog();

	//@name:	shader所在目录的路径
	std::shared_ptr<Entry> get(const std::string& name);
	void update();

private:
	class Impl;
	std::unique_ptr<Impl> impl;
};

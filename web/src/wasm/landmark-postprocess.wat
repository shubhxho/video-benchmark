(module
  (memory (export "memory") 1)
  (global $heap (mut i32) (i32.const 0))

  (func $ensure_capacity (param $required_end i32)
    (local $capacity_bytes i32)
    memory.size
    i32.const 65536
    i32.mul
    local.set $capacity_bytes
    block $done
      loop $grow
        local.get $capacity_bytes
        local.get $required_end
        i32.ge_u
        br_if $done
        i32.const 1
        memory.grow
        drop
        local.get $capacity_bytes
        i32.const 65536
        i32.add
        local.set $capacity_bytes
        br $grow
      end
    end)

  (func (export "reset")
    i32.const 0
    global.set $heap)

  (func (export "alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local $end i32)
    global.get $heap
    local.set $ptr
    local.get $ptr
    local.get $size
    i32.add
    local.set $end
    local.get $end
    call $ensure_capacity
    local.get $end
    global.set $heap
    local.get $ptr)

  (func $landmark_offset (param $ptr i32) (param $index i32) (result i32)
    local.get $ptr
    local.get $index
    i32.const 12
    i32.mul
    i32.add)

  (func (export "smoothLandmarks")
    (param $current_ptr i32)
    (param $previous_ptr i32)
    (param $count i32)
    (param $out_ptr i32)
    (local $i i32)
    (local $curr_offset i32)
    (local $prev_offset i32)
    (local $out_offset i32)
    (local $curr_x f32)
    (local $curr_y f32)
    (local $curr_v f32)
    (local $prev_x f32)
    (local $prev_y f32)
    (local $prev_v f32)
    (local $out_v f32)

    i32.const 0
    local.set $i

    block $done
      loop $loop
        local.get $i
        local.get $count
        i32.ge_u
        br_if $done

        local.get $current_ptr
        local.get $i
        call $landmark_offset
        local.set $curr_offset
        local.get $previous_ptr
        local.get $i
        call $landmark_offset
        local.set $prev_offset
        local.get $out_ptr
        local.get $i
        call $landmark_offset
        local.set $out_offset

        local.get $curr_offset
        f32.load
        local.set $curr_x
        local.get $curr_offset
        i32.const 4
        i32.add
        f32.load
        local.set $curr_y
        local.get $curr_offset
        i32.const 8
        i32.add
        f32.load
        local.set $curr_v

        local.get $prev_offset
        f32.load
        local.set $prev_x
        local.get $prev_offset
        i32.const 4
        i32.add
        f32.load
        local.set $prev_y
        local.get $prev_offset
        i32.const 8
        i32.add
        f32.load
        local.set $prev_v

        local.get $curr_v
        f32.const 0.01
        f32.gt
        if
          local.get $prev_v
          f32.const 0.01
          f32.gt
          if
            local.get $out_offset
            local.get $curr_x
            f32.const 0.72
            f32.mul
            local.get $prev_x
            f32.const 0.28
            f32.mul
            f32.add
            f32.store

            local.get $out_offset
            i32.const 4
            i32.add
            local.get $curr_y
            f32.const 0.72
            f32.mul
            local.get $prev_y
            f32.const 0.28
            f32.mul
            f32.add
            f32.store

            local.get $curr_v
            f32.const 0.85
            f32.mul
            local.get $prev_v
            f32.const 0.15
            f32.mul
            f32.add
            local.set $out_v

            local.get $out_offset
            i32.const 8
            i32.add
            local.get $out_v
            f32.const 1
            f32.min
            f32.store
          else
            local.get $out_offset
            local.get $curr_x
            f32.store
            local.get $out_offset
            i32.const 4
            i32.add
            local.get $curr_y
            f32.store
            local.get $out_offset
            i32.const 8
            i32.add
            local.get $curr_v
            f32.store
          end
        else
          local.get $prev_v
          f32.const 0.6
          f32.gt
          if
            local.get $out_offset
            local.get $prev_x
            f32.store
            local.get $out_offset
            i32.const 4
            i32.add
            local.get $prev_y
            f32.store
            local.get $out_offset
            i32.const 8
            i32.add
            local.get $prev_v
            f32.const 0.78
            f32.mul
            f32.store
          else
            local.get $out_offset
            f32.const 0
            f32.store
            local.get $out_offset
            i32.const 4
            i32.add
            f32.const 0
            f32.store
            local.get $out_offset
            i32.const 8
            i32.add
            f32.const 0
            f32.store
          end
        end

        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $loop
      end
    end)

  (func (export "meanVisibility")
    (param $ptr i32)
    (param $count i32)
    (result f32)
    (local $i i32)
    (local $sum f32)

    local.get $count
    i32.eqz
    if (result f32)
      f32.const 0
    else
      i32.const 0
      local.set $i
      f32.const 0
      local.set $sum

      block $done
        loop $loop
          local.get $i
          local.get $count
          i32.ge_u
          br_if $done
          local.get $sum
          local.get $ptr
          local.get $i
          call $landmark_offset
          i32.const 8
          i32.add
          f32.load
          f32.add
          local.set $sum
          local.get $i
          i32.const 1
          i32.add
          local.set $i
          br $loop
        end
      end

      local.get $sum
      local.get $count
      f32.convert_i32_u
      f32.div
    end)

  (func (export "indexedVisibility")
    (param $ptr i32)
    (param $landmark_count i32)
    (param $indices_ptr i32)
    (param $index_count i32)
    (result f32)
    (local $i i32)
    (local $index i32)
    (local $valid i32)
    (local $sum f32)

    i32.const 0
    local.set $i
    i32.const 0
    local.set $valid
    f32.const 0
    local.set $sum

    block $done
      loop $loop
        local.get $i
        local.get $index_count
        i32.ge_u
        br_if $done

        local.get $indices_ptr
        local.get $i
        i32.const 4
        i32.mul
        i32.add
        i32.load
        local.set $index

        local.get $index
        local.get $landmark_count
        i32.lt_u
        if
          local.get $sum
          local.get $ptr
          local.get $index
          call $landmark_offset
          i32.const 8
          i32.add
          f32.load
          f32.add
          local.set $sum
          local.get $valid
          i32.const 1
          i32.add
          local.set $valid
        end

        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $loop
      end
    end

    local.get $valid
    i32.eqz
    if (result f32)
      f32.const 0
    else
      local.get $sum
      local.get $valid
      f32.convert_i32_u
      f32.div
    end)
)
